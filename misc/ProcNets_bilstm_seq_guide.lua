require 'nn'
require 'csvigo'

local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTMCell = require 'misc.LSTMCell'
require 'misc.LSTM'
require 'misc.ScoreMaxpool'
require 'misc.TemporalSegmentation_bilstm_seq'

-------------------------------------------------------------------------------
-- ProcNets Core Part
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.ProcNets', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  local dropout = utils.getopt(opt, 'dropout', 0)

  -- options for temporal segmentation layer
  local kernelTsLow = utils.getopt(opt, 'KTL', 3)
  local kernelTsUp = utils.getopt(opt, 'KTU', 123)
  local kernelTsInterval = utils.getopt(opt, 'KTS', 8)
  self.clip_number = utils.getopt(opt, 'CN', 15)

  -- options for ProcNets
  self.frames_per_video = utils.getopt(opt, 'frames_per_video')
  local mp_scale_h = utils.getopt(opt, 'mp_scale_h', 8)
  local mp_scale_w = utils.getopt(opt, 'mp_scale_w', 5)
  self.gradcheck = utils.getopt(opt, 'gradcheck', false)

  ---------------------------------------------------------------------------------
  ------------------------- ProcNets model construction ---------------------------
  ---------------------------------------------------------------------------------
  -- 1. context-aware feature layer
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- feature data
  table.insert(inputs, nn.Identity()()) -- inversed feature data
  local lstm_f = nn.LSTM(self.input_encoding_size, self.input_encoding_size)(inputs[1])
  local lstm_b = nn.LSTM(self.input_encoding_size, self.input_encoding_size)(inputs[2])
  local concat = nn.JoinTable(3)({inputs[1], lstm_f, lstm_b})
  local mlp = nn.Sequential()
  mlp:add(nn.Bottle(nn.Linear(3*self.input_encoding_size, self.rnn_size)))
  mlp:add(nn.ReLU())
  local output = mlp(concat)
  self.core1 = nn.gModule(inputs, {output})

  -- 2. temporal convolution layer
  self.ts = nn.TemporalSegmentation({kernelTsLow, kernelTsUp, kernelTsInterval}, {self.rnn_size, self.frames_per_video}, self.clip_number)

  -- 3a. max pooling on the proposed scores
  self.maxpool = nn.ScoreMaxpool(mp_scale_w,mp_scale_h,self.gradcheck)

  -- 3b. 1-layer lstm for sequential modeling
  self.clip_prop_encoding = self.frames_per_video*((kernelTsUp-kernelTsLow)/kernelTsInterval+1)/mp_scale_h/mp_scale_w
  self.core2 = LSTMCell.lstm_cell(self.clip_prop_encoding*3, self.clip_prop_encoding+1, self.clip_prop_encoding, 1, dropout)
  -- lookup_table for location embedding, seg_lin for segment content embedding
  self.lookup_table = nn.LookupTable(self.clip_prop_encoding+1, self.clip_prop_encoding)
  self.seg_lin = nn.Linear(self.input_encoding_size, self.clip_prop_encoding)

  self:_createInitState(1) -- will be lazily resized later during forward passes
  ---------------------------------------------------------------------------------
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  --initial state for second lstm layer 
  if not self.init_state2 then self.init_state2 = {} end
  for h=1,2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state2[h] then
      if self.init_state2[h]:size(1) ~= batch_size then
        self.init_state2[h]:resize(batch_size, self.clip_prop_encoding):zero() -- expand the memory
      end
    else
      self.init_state2[h] = torch.zeros(batch_size, self.clip_prop_encoding)
    end
  end
  self.num_state2 = #self.init_state2
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the ProcNets')
  self.clones2 = {self.core2}
  self.lookup_tables = {self.lookup_table}
  self.seg_lins = {self.seg_lin}
  for t=2,self.clip_number+1 do
    self.clones2[t] = self.core2:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
    self.seg_lins[t] = self.seg_lin:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end

function layer:getModulesList()
  return {self.core1, self.core2, self.ts.tc, self.lookup_table, self.seg_lin}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core1:parameters()
  local p2,g2 = self.core2:parameters()
  local p3,g3 = self.ts:parameters()
  local p4,g4 = self.lookup_table:parameters()
  local p5,g5 = self.seg_lin:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end  
  for k,v in pairs(p3) do table.insert(params, v) end
  for k,v in pairs(p4) do table.insert(params, v) end
  for k,v in pairs(p5) do table.insert(params, v) end 

  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end
  for k,v in pairs(g3) do table.insert(grad_params, v) end
  for k,v in pairs(g4) do table.insert(grad_params, v) end
  for k,v in pairs(g5) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory
  return params, grad_params
end

function layer:training()
  if self.clones2 == nil or self.lookup_tables == nil or self.seg_lins == nil then self:createClones() end -- create these lazily if needed
  self.core1:training()
  for k,v in pairs(self.clones2) do v:training() end
  self.ts:training()
  for k,v in pairs(self.lookup_tables) do v:training() end
  for k,v in pairs(self.seg_lins) do v:training() end
end

function layer:evaluate()
  if self.clones2 == nil or self.lookup_tables == nil or self.seg_lins == nil then self:createClones() end -- create these lazily if needed
  self.core1:evaluate()
  for k,v in pairs(self.clones2) do v:evaluate() end
  self.ts:evaluate()
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
  for k,v in pairs(self.seg_lins) do v:evaluate() end
end

-- Inference. Use greedy sampling
function layer:sample(input, opt)
  -- check the size of the input
  assert(input:size(1)==self.frames_per_video)
  assert(input:size(2)==self.input_encoding_size) 

  -- if beam_size > 1, call the sample_beam function 
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  
  local batch_size = input:size(1)/self.frames_per_video
  self:_createInitState(batch_size)
  local imgs = input:view(batch_size, input:size(1), input:size(2))
  local imgs_inverse = torch.Tensor(imgs:size()):type(imgs:type())
  for i=1,imgs:size(2) do
     imgs_inverse[{1,i}] = imgs[{1,self.frames_per_video+1-i}]
  end
  -- the output of the first-layer lstm
  self.lstm_output1 = self.core1:forward({imgs, imgs_inverse})
  -- forward through the temporal segmentation layer
  self.ts_output = self.ts:forward(self.lstm_output1:view(self.frames_per_video, self.rnn_size))
  -- forward through the proposal score max pooling layer
  local score_mp = self.maxpool:forward(self.ts_output)
  -- forward through the sequencial modeling module
  -- score_mp is a two-tuple table of {score, boundaries}
  assert(score_mp[1]:size(1)*score_mp[1]:size(2) == self.clip_prop_encoding)
  local tran_score_mp = score_mp[1]:transpose(1,2):clone()
  local score_feat = tran_score_mp:view(1,-1):clone()
  local prop_index = {}

  if beam_size > 1 then
    local prop_index_beam = self:sample_beam({score_feat,input,score_mp}, opt)
    prop_index_beam = prop_index_beam:view(-1)
    local prop_index_all = torch.totable(prop_index_beam)
    for i=1,#prop_index_all do -- remove the zeros at the end
      if prop_index_all[i] ~= 0 then
          prop_index[i] = prop_index_all[i]
      end
    end
  else
    local state2 = {[0] = self.init_state2}
    local lstm2_input = {}
    -- csvigo.save{path = 'prop_score.csv', data = torch.totable(self.lookup_table.weight)}
  
    for t=1,self.clip_number+1 do
      local xt,it,emb,itseq, embseq
      if t==1 then
        it = torch.LongTensor(1):fill(self.clip_prop_encoding+1)
        emb = self.lookup_table:forward(it)
        itseq = torch.mean(input,1)
        local tmpseq = self.seg_lin:forward(itseq)
        embseq = tmpseq:clone()
        xt = torch.cat(torch.cat(score_feat,emb),embseq)
      else
        it = torch.Tensor(1):fill(prop_index[t-1]):type(input:type())
        emb = self.lookup_table:forward(it)
        -- the clip embedding correspond to the index
        local column = math.floor((prop_index[t-1]-1)/(score_mp[1]:size(1)))+1
        local row = prop_index[t-1]-(column-1)*score_mp[1]:size(1)
        local feat_block = input:sub(math.max(score_mp[2][{1,row,column}],1), math.min(score_mp[2][{2,row,column}],input:size(1)),1,self.input_encoding_size):clone()
        itseq = torch.mean(feat_block,1)
        -- forward the clip embedding through the linear layer
        local tmpseq = self.seg_lin:forward(itseq)
        embseq = tmpseq:clone()
        xt = torch.cat(torch.cat(score_feat,emb),embseq)
      end
     
      lstm2_input[t] = {xt,unpack(state2[t-1])}
      local out = self.core2:forward(lstm2_input[t])
      local lstm2_output = out[self.num_state2+1]
      local val, ind = torch.max(lstm2_output,2)
      prop_index[t] = ind[{1,1}]
      state2[t] = {}
      for i=1,self.num_state2 do table.insert(state2[t], out[i]) end
      if prop_index[t] == lstm2_output:size(2) then break end  -- stop bit
    end
  end

  -- from index to actual segment boundary
  local clip_prop = torch.LongTensor(#prop_index-1,3):type(input:type())
  for i=1,#prop_index-1 do
    local column = math.floor((prop_index[i]-1)/score_mp[1]:size(1))+1
    local row = prop_index[i] - (column-1)*score_mp[1]:size(1)
    clip_prop[{i,1}] = score_mp[2][{1,row,column}]
    clip_prop[{i,2}] = score_mp[2][{2,row,column}]
    clip_prop[{i,3}] = score_mp[1][{row,column}] -- proposal score
  end
  self.output = {self.ts_output[1], self.ts_output[2], clip_prop, score_mp[1], score_mp[2]}
  return self.output
end

-- beam search adapted from neuraltalk2
function layer:sample_beam(input_table, opt)
  local imgs = input_table[1]
  local input = input_table[2]
  local score_mp = input_table[3]
  local beam_size = utils.getopt(opt, 'beam_size', 2)
  local batch_size, feat_dim = imgs:size(1), imgs:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream
  assert(beam_size <= self.clip_prop_encoding+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')
  local seq = torch.LongTensor(self.clip_number, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.clip_number, batch_size)
  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do
    -- create initial states for all beams
    self:_createInitState(beam_size)
    local state = self.init_state2
    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.clip_number, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.clip_number, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    local imgk = imgs[{ {k,k} }]:expand(beam_size, feat_dim) -- k'th image feature expanded out
    for t=1,self.clip_number+1 do
      local xt, it, sampleLogprobs
      local new_state
      if t == 1 then
        it = torch.LongTensor(beam_size):fill(self.clip_prop_encoding+1)
        local emb = self.lookup_table:forward(it)
        local itseq = torch.mean(input,1)
        local tmpseq = self.seg_lin:forward(itseq)
        local embseq = torch.expand(tmpseq,beam_size,tmpseq:size(2))
        xt = torch.cat(torch.cat(imgk,emb),embseq)
      else
        --[[
          perform a beam merge. that is,
          for every previous beam we now many new possibilities to branch out
          we need to resort our beams to maintain the loop invariant of keeping
          the top beam_size most likely sequences.
        ]]--
        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 2 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 2 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-2}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-2}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 2 then
            beam_seq[{ {1,t-2}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-2}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-1, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-1, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.clip_prop_encoding+1 or t == self.clip_number+1 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end
        
        -- encode as vectors
        it = beam_seq[t-1]
        local emb = self.lookup_table:forward(it)
        local itseq = torch.Tensor(beam_size,self.input_encoding_size):type(emb:type())
        for cc=1,beam_size do
          local column = math.floor((it[cc]-1)/(score_mp[1]:size(1)))+1
          local row = it[cc]-(column-1)*score_mp[1]:size(1)
          if column > score_mp[1]:size(2) then -- special bit
              itseq[cc] = torch.mean(input,1)
          else
              local feat_block = input:sub(math.max(score_mp[2][{1,row,column}],1), math.min(score_mp[2][{2,row,column}],input:size(1)),1,self.input_encoding_size):clone()
              itseq[cc] = torch.mean(feat_block,1)
          end
        end
        -- forward the clip embedding through the linear layer
        local tmpseq = self.seg_lin:forward(itseq)
        local embseq = tmpseq:clone()
        xt = torch.cat(torch.cat(imgk,emb),embseq)
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

      local inputs = {xt,unpack(state)}
      local out = self.core2:forward(inputs)
      logprobs = out[self.num_state2+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state2 do table.insert(state, out[i]) end
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq
end

-- function: find the nearest segment in the max-pooled proposals
local function nearest_segment(boundary, target)
-- input source segments (2*k/mp_scale*L/mp_scale tensor) and target segment (table).
-- output the index of the nearest segment (index by column order)
  local result = torch.LongTensor(#target,1):type(boundary:type())
  -- transpose boundary
  local tran_boundary = boundary:transpose(2,3):clone()

  for i=1,#target do
    local min_dis = 500^2
    local index = 0
    for j=1,boundary:size(2)*boundary:size(3) do
       local low = tran_boundary:storage()[j]
       local high = tran_boundary:storage()[j+boundary:size(2)*boundary:size(3)]
       local dis = (target[i][1]-low)^2+(target[i][2]-high)^2
       if dis < min_dis then
           index = j
           min_dis = dis
       end
    end
    result[i] = index
  end
  return result
end

--[[
Input: 1) context-aware feature, 2) ground-truth segments
Output: 1) temp conv output on the proposal score, 2) temp conv output on the proposal boundaries, 3) encoded ground-truth segments (index), 4) the softmax outputs of sequential modeling LSTM
--]]
function layer:updateOutput(input_table)
  local input = input_table[1]
  local segments = input_table[2]

  -- check the size of the input
  assert(input:size(1)==self.frames_per_video)
  assert(input:size(2)==self.input_encoding_size) 
 
  if self.clones2 == nil or self.lookup_tables == nil then self:createClones() end -- lazily create clones on first forward pass
  local batch_size = input:size(1)/self.frames_per_video
  self:_createInitState(batch_size)
  local imgs = input:view(batch_size, input:size(1), input:size(2))
  local imgs_inverse = torch.Tensor(imgs:size()):type(imgs:type())
  -- the output of the first-layer lstm
  for i=1,imgs:size(2) do
     imgs_inverse[{1,i}] = imgs[{1,self.frames_per_video+1-i}]
  end

  self.lstm_output1 = self.core1:forward({imgs, imgs_inverse})

  -- forward through the temporal segmentation layer
  self.ts_output = self.ts:forward(self.lstm_output1:view(self.frames_per_video, self.rnn_size))
  -- forward through the proposal score max pooling layer
  local score_mp = self.maxpool:forward(self.ts_output)
  -- forward through the sequence learning module
  -- score_mp is a two-tuple table of {score, boundaries}
  assert(score_mp[1]:size(1)*score_mp[1]:size(2) == self.clip_prop_encoding)
  local tran_score_mp = score_mp[1]:transpose(1,2):clone()
  local score_feat = tran_score_mp:view(1,-1):clone()
  local gt_seg_enc = nearest_segment(score_mp[2], segments)

  -- we skip the `start` token.
  self.lstm2_input = {}
  self.state2 = {[0] = self.init_state2}
  -- the last output segment should be stop bit
  local lstm2_output=torch.Tensor(#segments+1,self.clip_prop_encoding+1):type(input:type())
  self.seg_lins_inputs = torch.Tensor(#segments+1,self.input_encoding_size):type(input:type())
  self.seg_lins_index = {}
  for t=1,#segments+1 do
    local xt
    -- concatenate the three features and form the LSTM input
    if t == 1 then
      local it = torch.LongTensor(1):fill(self.clip_prop_encoding+1)
      local tmp = self.lookup_tables[t]:forward(it)
      local emb = tmp:clone()
      local itseq = torch.mean(input,1)
      self.seg_lins_inputs[t] = itseq:clone()
      local tmpseq = self.seg_lins[t]:forward(itseq)
      local embseq = tmpseq:clone()
      xt = torch.cat(torch.cat(score_feat,emb),embseq)
    else
      local it = gt_seg_enc[t-1]:clone()
      local tmp = self.lookup_tables[t]:forward(it)
      local emb = tmp:clone()

      local column = math.floor((gt_seg_enc[t-1][1]-1)/(score_mp[1]:size(1)))+1
      local row = gt_seg_enc[t-1][1]-(column-1)*score_mp[1]:size(1)
      local feat_block = input:sub(math.max(score_mp[2][{1,row,column}],1), math.min(score_mp[2][{2,row,column}],input:size(1)),1,self.input_encoding_size):clone()
      local itseq = torch.mean(feat_block,1)
      self.seg_lins_inputs[t] = itseq:clone()
      self.seg_lins_index[t] = {math.floor(math.max(score_mp[2][{1,row,column}],1)), math.floor(math.min(score_mp[2][{2,row,column}],input:size(1)))}
      assert(feat_block:size(1)==self.seg_lins_index[t][2]-self.seg_lins_index[t][1]+1)
      local tmpseq = self.seg_lins[t]:forward(itseq)
      local embseq = tmpseq:clone()
      xt = torch.cat(torch.cat(score_feat,emb),embseq)
    end
 
    -- forward through LSTM
    self.lstm2_input[t] = {xt,unpack(self.state2[t-1])}
    local out = self.clones2[t]:forward(self.lstm2_input[t])
    lstm2_output[t] = out[self.num_state2+1] -- last element is the output vector
    self.state2[t] = {} -- the rest is state
    for i=1,self.num_state2 do table.insert(self.state2[t], out[i]) end
  end

  self.lookup_tables_inputs = gt_seg_enc:clone()
  self.output = {self.ts_output[1],self.ts_output[2],gt_seg_enc,lstm2_output}
  return self.output
end

function layer:updateGradInput(input_table, gradOutput)
  local input = input_table[1]
  local segments = input_table[2]
  local mp_scale_h = self.maxpool.kernelsize_h
  local mp_scale_w = self.maxpool.kernelsize_w
  -- gradients on Propsal Vector
  local dscore_feat
  -- gradients from Segment Content feature to input
  local dinput_gt = torch.Tensor(input:size()):zero():type(input:type())

  -- backward through sequential modeling lstm
  local dstate = {[#segments+1] = self.init_state2}
  for t=#segments+1,1,-1 do
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[4][t]:view(1,-1))

    local dinputs = self.clones2[t]:backward(self.lstm2_input[t],dout)
    -- split the gradient to xt and to state
    local dxt = dinputs[1] -- first element is the input vector
    dstate[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state2+1 do table.insert(dstate[t-1],dinputs[k]) end

    -- backprop through the three concatenated features
    if t == 1 then
      dscore_feat = dscore_feat + dxt:sub(1,1,1,self.clip_prop_encoding):clone()
      local it = torch.LongTensor(1):fill(self.clip_prop_encoding+1)
      local demb = dxt:sub(1,1,self.clip_prop_encoding+1,self.clip_prop_encoding*2):clone()
      self.lookup_tables[t]:backward(it,demb)
      -- derivative for the segment content feature
      local itseq = self.seg_lins_inputs[t]:view(1,-1):clone()
      local dembseq = dxt:sub(1,1,self.clip_prop_encoding*2+1,self.clip_prop_encoding*3):clone()
      local dlookup = self.seg_lins[t]:backward(itseq,dembseq)
      dinput_gt:add(torch.expand(dlookup,input:size(1),input:size(2))/input:size(1))
    else
      if t == #segments+1 then
        dscore_feat = dxt:sub(1,1,1,self.clip_prop_encoding):clone()
      else
        dscore_feat = dscore_feat + dxt:sub(1,1,1,self.clip_prop_encoding):clone()
      end
      local it = self.lookup_tables_inputs[t-1]:clone()
      local demb = dxt:sub(1,1,self.clip_prop_encoding+1,self.clip_prop_encoding*2):clone()
      self.lookup_tables[t]:backward(it,demb)
      -- derivative for the segment content feature 
      local itseq = self.seg_lins_inputs[t]:view(1,-1):clone()
      local dembseq = dxt:sub(1,1,self.clip_prop_encoding*2+1,self.clip_prop_encoding*3):clone()
      local dlookup = self.seg_lins[t]:backward(itseq,dembseq)
      local low = self.seg_lins_index[t][1]
      local high = self.seg_lins_index[t][2]
      dinput_gt:sub(low,high,1,input:size(2)):add(torch.expand(dlookup,(high-low+1),input:size(2))/(high-low+1))
    end
  end

  dscore_feat:resize(self.ts_output[1]:size(2)/mp_scale_w,self.ts_output[1]:size(1)/mp_scale_h)
  local tran_dscore_feat = dscore_feat:transpose(1,2):clone()

  -- backward through max pooling layer
  local dmp_in = self.maxpool:backward(self.ts_output,{tran_dscore_feat,torch.Tensor()})

  -- the gradients on the temp conv outputs
  local dts_output = {}
  dts_output[1] = dmp_in[1] + gradOutput[1]
  dts_output[2] = gradOutput[2]

  -- backward through the temp conv layer
  local dlstm_output1 = self.ts:backward(self.lstm_output1:view(self.frames_per_video, self.rnn_size), dts_output)
  assert(dlstm_output1:nDimension() == 2)
  local batch_size = input:size(1)/self.frames_per_video
  local imgs = input:view(batch_size, input:size(1), input:size(2))
  local imgs_inverse = torch.Tensor(imgs:size()):type(imgs:type())
  for i=1,imgs:size(2) do
     imgs_inverse[{1,i}] = imgs[{1,self.frames_per_video+1-i}]
  end
  
  -- backward through the bi-lstm layer
  local dlstm_input1 = self.core1:backward({imgs, imgs_inverse}, dlstm_output1:view(1, self.frames_per_video, self.rnn_size))

  local dinput = dlstm_input1[1]:view(self.frames_per_video, self.input_encoding_size)
  local dinput_inverse = dlstm_input1[2]:view(self.frames_per_video, self.input_encoding_size)
  for i=1,dinput:size(1) do
     dinput[i]:add(dinput_inverse[self.frames_per_video+1-i])
  end
  dinput:add(dinput_gt) -- add derivative w.r.t. gt segment guidance
  self.gradInput = {dinput,{}}
  return self.gradInput
end

-------------------------------------------------------------------------------
-- ProcNets Criterion
-------------------------------------------------------------------------------
local function iou(c1, c2)
    intersection = math.max(0, math.min(c1[2], c2[2])-math.max(c1[1], c2[1]))
    if intersection == 0 then
        return 0
    else
        union = math.max(c1[2], c2[2]) - math.min(c1[1], c2[1])
        return intersection/union
    end
end

local function smoothl1loss(x)
    if torch.abs(x)<1 then
        return 0.5*torch.pow(x,2)
    else
        return torch.abs(x)-0.5
    end
end

local function dsmoothl1loss(x)
    if torch.abs(x)<1 then
        return x
    elseif x>=1 then
        return 1
    else
        return -1
    end
end

-- function for shuffling tables
local function shuffleTable(t)
    local rand = math.random
    assert(t, "shuffleTable() expected a table, got nil")
    local iterations = #t
    local j
    for i = iterations, 2, -1 do
        j = rand(i)
        t[i], t[j] = t[j], t[i]
    end
end

local crit, parent = torch.class('nn.ProcNetsCriterion', 'nn.Criterion')
function crit:__init(kernelinfo,train_sample, gradcheck)
  parent.__init(self)
  self.KTL = kernelinfo[1]
  self.KTU = kernelinfo[2]
  self.KTS = kernelinfo[3]
  self.train_sample = train_sample or 4  -- 4 for grad check
  if gradcheck == nil then
      self.gradcheck = true  -- true for grad check
  else
      self.gradcheck = gradcheck
  end
end

function crit:updateOutput(input_table, segments)

  -- cross entropy loss should put here
  -- input: k*L    label: temporal segments
  -- output: cross-entropy loss
  local input = input_table[1]
  local boundary = input_table[2]
  local gt_seg_enc = input_table[3]
  local seg_prob = input_table[4]
  local positive_thre = 0.8
  local negative_thre = 0.2
  local sample_num = self.train_sample
  local seg_num = #segments
  local loss = 0 -- classification loss
  local loss_reg = 0 -- regression loss
  local loss_alpha_r = 1 -- discount for regression loss
  local loss_seq = 0 -- sequence modeling loss
  local loss_alpha_s = 1 -- discount for sequence modeling loss
  local positive_counter = 0
  local negative_counter = 0
  local positive_clip = {}
  local negative_clip = {}

  self.gradInput = {}
  self.gradInput[1] = torch.Tensor(input:size()):zero():type(input:type())
  self.gradInput[2] = torch.Tensor(boundary:size()):zero():type(boundary:type())
  self.gradInput[4] = torch.Tensor(seg_prob:size()):zero():type(seg_prob:type())
  self.gradInput[3] = torch.Tensor() -- empty tensor

  -- random pick positive/negative samples
  local rand_order_i = torch.rand(input:size(1))
  local v_i, ind_i = torch.sort(rand_order_i)
  local rand_order_j = torch.rand(input:size(2))
  local v_j, ind_j = torch.sort(rand_order_j)

  for ii=1, input:size(1) do
      for jj=1, input:size(2) do
          if not self.gradcheck then i = ind_i[ii] j = ind_j[jj] else i = ii j = jj end
          local clip_low = boundary[{1,i,j}]
          local clip_high = boundary[{2,i,j}]
          local positive_flag = 0
          local negative_flag = 1
          -- clip should be inside the video
          if clip_low > 0 and  clip_high <= input:size(2) then
              local rand_order = torch.rand(seg_num)
              local v, ind = torch.sort(rand_order)
              for kk=1, seg_num do
                  if not self.gradcheck then k = ind[kk] else k = kk end
                  local clip_iou = iou({clip_low,clip_high},segments[k])
                  if clip_iou >= positive_thre then
                      -- compute smooth l1 loss for boundaries, not offsets (doesn't make much difference on performance)
                      loss_reg = loss_reg + smoothl1loss(clip_low-segments[k][1]) + smoothl1loss(clip_high-segments[k][2])
                      self.gradInput[2][{1,i,j}] = dsmoothl1loss(clip_low-segments[k][1])
                      self.gradInput[2][{2,i,j}] = dsmoothl1loss(clip_high-segments[k][2])
                      positive_flag = 1
                      break
                  elseif clip_iou >= negative_thre then
                      negative_flag = 0
                  end
              end
              if positive_flag==1 then
                  positive_counter = positive_counter + 1
                  positive_clip[positive_counter] = {i,j}
              elseif negative_flag==1 then
                  negative_counter = negative_counter + 1
                  negative_clip[negative_counter] = {i,j}
              end
          end
          -- break if get enough training data
          if positive_counter>=sample_num/2 and negative_counter>=sample_num/2 then break end
      end
      if positive_counter>=sample_num/2 and negative_counter>=sample_num/2 then break end
  end

  assert(positive_counter+negative_counter>=sample_num)
  print('+', positive_counter, '-', negative_counter)

  -- classification loss
  if positive_counter>0 then
      for k=1, math.min(sample_num/2, positive_counter) do
          local ib = positive_clip[k][1]
          local jb = positive_clip[k][2]
          loss = loss - torch.log(input[ib][jb])
          self.gradInput[1][{ib,jb}] = -1/input[ib][jb]
      end
  end
  for k=1, sample_num-math.min(sample_num/2, positive_counter) do
      local ib = negative_clip[k][1]
      local jb = negative_clip[k][2]
      loss = loss - torch.log(1-input[ib][jb])
      self.gradInput[1][{ib,jb}] = 1/(1-input[ib][jb])
  end
  loss = loss/sample_num
  self.gradInput[1] = self.gradInput[1]/sample_num

  -- regression loss, count all the positive proposals (not just positive samples)
  if positive_counter>0 then
      loss_reg = loss_reg*loss_alpha_r/positive_counter
      self.gradInput[2] = self.gradInput[2]:div(positive_counter/loss_alpha_r)
  end
  
  -- segment regression loss
  assert(gt_seg_enc:size(1)+1 == seg_prob:size(1))
  assert(gt_seg_enc:size(1) == #segments)
  for i=1,gt_seg_enc:size(1) do
      loss_seq = loss_seq - seg_prob[{i,gt_seg_enc[i][1]}]
      self.gradInput[4][{i,gt_seg_enc[i][1]}] = -1
  end
  -- the stop bit
  loss_seq = loss_seq - seg_prob[{-1,-1}]
  self.gradInput[4][{-1,-1}] = -1
  
  self.gradInput[4]:div((gt_seg_enc:size(1)+1)/loss_alpha_s)
  loss_seq = loss_seq*loss_alpha_s/(gt_seg_enc:size(1)+1)

  self.output = loss + loss_reg + loss_seq
  return self.output
end

function crit:updateGradInput(input_table, segments)
  return self.gradInput
end
