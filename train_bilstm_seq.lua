require 'torch'
require 'nn'
require 'nngraph'
require 'csvigo'

require 'misc.DataLoader'
require 'misc.DataLoaderVal'
require 'misc.optim_updates'
require 'misc.ProcNets_bilstm_seq_guide'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train and validate a Procedure Segmentation model')
cmd:text()
cmd:text('Options')

-- Data input/ouput settings, change to your path
cmd:option('-image_folder', '/z/home/luozhou/YouCookII/features/feat_dat_500/', 'root directory of image features')
cmd:option('-train_data_folder', 'train_frame_feat', 'training data folder')
cmd:option('-val_data_folder', 'val_frame_feat', 'training data folder')
cmd:option('-ann_file', '/z/home/luozhou/YouCookII/annotations/youcookii_annotations_trainval.json', 'the raw annotation .json files')
cmd:option('-train_vidinfo_file', '/z/home/luozhou/YouCookII/annotations/duration_frame/train_duration_totalframe.csv', 'file that stores duration and total frame info')
cmd:option('-val_vidinfo_file', '/z/home/luozhou/YouCookII/annotations/duration_frame/val_duration_totalframe.csv', 'file that stores duration and total frame info')
cmd:option('-checkpoint_path', '/z/home/luozhou/dataset/YouCookII_checkpoint/', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')

-- Model settings: General
cmd:option('-rnn_size', 512, 'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size', 512, 'the encoding size of the image feature')
cmd:option('-emb_mode', 'bilstm', 'the embedding mode for input resnet feature')
cmd:option('-train_sample', 200, 'total number of positive+negative training samples (2*U)')
cmd:option('-clip_number', 16, 'the maximum number of clips per video')

-- Model settings: for temp conv and max pooling
cmd:option('-frames_per_video', 500, 'number of frames sampled in one video')
cmd:option('-KTL', 3, 'smallest kernel size for temporal conv')
cmd:option('-KTU', 123, 'largest kernel size for temporal conv')
cmd:option('-KTS', 8, 'kernel size interval for temporal conv')
cmd:option('-mp_scale_h', 8, 'proposal score max pooling kernel height')
cmd:option('-mp_scale_w', 5, 'proposal score max pooling kernel width')

-- Optimization: General
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-batch_size',1,'what is the batch size in number of videos per batch? Now we only support batch size equals to 1...')
cmd:option('-grad_clip',1,'clip gradients at this value')
cmd:option('-drop_prob_lm', 0.5, 'strength of dropout in the Language Model RNN')

-- Optimization: for ProcNets
cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-5,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 100000, 'every how many iterations thereafter to drop LR by half?')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')
cmd:option('-lm_weight_decay', 0.001, 'L2 weight decay just for the LM')

-- Evaluation/Checkpointing
cmd:option('-val_images_use', -1, 'how many videos to use when periodically evaluating the validation loss? -1 = all')
cmd:option('-beam_size', 1, 'used when sample_max = 1, indicates number of beams in beam search.')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-losses_log_every', 50, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-gradcheck', false, 'an option for grad_check script only, do not change it here')
cmd:option('-vis', false, 'if true, save segment visualizations to folder')
cmd:option('-write_output', false, 'if true, output the proposal results to a .json file')

cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

assert(opt.batch_size == 1) -- for now, only support batch_size=1
opt.gradcheck = opt.gradcheck == "true"
opt.vis = opt.vis == "true"
opt.write_output = opt.write_output == "true"

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader, loader_val
assert(string.len(opt.image_folder) > 0)
loader = DataLoader{folder_path = opt.image_folder, split = opt.train_data_folder, -- for training
                       ann_file = opt.ann_file, train_vidinfo_file = opt.train_vidinfo_file}
loader_val = DataLoaderVal{folder_path = opt.image_folder, split = opt.val_data_folder, -- for validation
                       ann_file = opt.ann_file, val_vidinfo_file = opt.val_vidinfo_file}

local protos = {}

if string.len(opt.start_from) > 0 then
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos

  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do
    net_utils.unsanitize_gradients(v)
  end

  protos.crit = nn.ProcNetsCriterion({protos.lm.ts.kernel_low, protos.lm.ts.kernel_high, protos.lm.ts.kernel_interval}, opt.train_sample, opt.gradcheck) -- not in checkpoints, create manually
else
  local lmOpt = {}
  lmOpt.input_encoding_size = opt.input_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.dropout = opt.drop_prob_lm
  lmOpt.frames_per_video = opt.frames_per_video -- number of unrolled LSTMs
  lmOpt.batch_size = opt.batch_size
  lmOpt.CN = opt.clip_number
  lmOpt.KTL = opt.KTL
  lmOpt.KTU = opt.KTU
  lmOpt.KTS = opt.KTS
  lmOpt.mp_scale_h = opt.mp_scale_h
  lmOpt.mp_scale_w = opt.mp_scale_w
  -- create the ProcNets model
  protos.lm = nn.ProcNets(lmOpt)
  protos.crit = nn.ProcNetsCriterion({opt.KTL, opt.KTU, opt.KTS}, opt.train_sample, opt.gradcheck)
end

-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local params, grad_params = protos.lm:getParameters()
print('total number of parameters in LM: ' , params:nElement())
assert(params:nElement() == grad_params:nElement())

local thin_lm = protos.lm:clone()
thin_lm.ts.tc:share(protos.lm.ts.tc, 'weight', 'bias') -- params sharing, we only want to store weights/bias
thin_lm.core1:share(protos.lm.core1, 'weight', 'bias')
thin_lm.core2:share(protos.lm.core2, 'weight', 'bias')
thin_lm.lookup_table:share(protos.lm.lookup_table, 'weight', 'bias')
thin_lm.seg_lin:share(protos.lm.seg_lin, 'weight', 'bias')

local lm_modules = thin_lm:getModulesList()
for k,v in pairs(lm_modules) do print(k, v) net_utils.sanitize_gradients(v) end

-- create clones and ensure parameter sharing. we have to do this
-- all the way here at the end because calls such as :cuda() and
-- :getParameters() reshuffle memory around!
protos.lm:createClones()

collectgarbage()

-- function: compute IoU or Jacc for two clip inputs
-- c1 and c2 can be either table or tensor
local function iou(c1, c2)
    intersection = math.max(0, math.min(c1[2], c2[2])-math.max(c1[1], c2[1]))
    if intersection == 0 then
        return 0
    else
        union = math.max(c1[2], c2[2]) - math.min(c1[1], c2[1])
        return  intersection/union
    end
end

local function jacc(c1, c2)
    -- c1 is gt, c2 is pred
    intersection = math.max(0, math.min(c1[2], c2[2])-math.max(c1[1], c2[1]))
    if intersection == 0 then
        return 0
    else
        return  intersection/(c2[2]-c2[1])
    end
end

-- function: visualize the segments
local function visSegment(gt, prop, file_path)
    -- background
    local linewidth = 10 -- fixed
    local vis = image.drawRect(torch.Tensor(3,50,500):fill(1.0), 6, 15, 495, 15, {lineWidth = linewidth, color = {211, 211, 211}})
    vis = image.drawRect(vis, 6, 35, 495, 35, {lineWidth = linewidth, color = {211, 211, 211}})
    -- color: red, orange
    local color_tab = {{255,0,0},{255,165,0},{255,255,0},{0,255,0},{51,153,255},{0,0,153},{102,0,204},{},{},{},{},{},{},{},{}}
    -- segments
    -- print(gt)
    -- print(prop)
    for i=1,#gt do
        vis = image.drawRect(vis, math.min(math.max(gt[i][1]+5,6),495), 15 ,
                             math.max(math.min(gt[i][2]-5,495),6), 15, {lineWidth = linewidth, color = color_tab[(i-1)%7+1]})
    end
    for i=1,prop:size(1) do
        vis = image.drawRect(vis, math.max(prop[i][1]+5,6),35, math.min(prop[i][2]-5,495), 
                             35, {lineWidth = linewidth, color = color_tab[(i-1)%7+1]})
    end
    -- add text
    local txt = image.drawText(torch.Tensor(3,50,50):fill(1.0), "Ground  truth", 1, 5, {color = {0,0,0}, size = 1})
    txt = image.drawText(txt, "Proposed", 1, 30, {color = {0,0,0}, size = 1})
    -- concatenate   
    vis = torch.cat(txt, vis, 3)
    image.save(path.join('./vis', string.sub(file_path,-16,-6)..'-'..string.sub(file_path,-4,-1)..'.jpg'), vis)
end

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(evalopt)
  local verbose = utils.getopt(evalopt, 'verbose', true)
  local val_images_use = utils.getopt(evalopt, 'val_images_use', true)
  protos.lm:evaluate()
  loader_val:resetIterator() -- rewind iteator back to first datapoint in the split

  local n = 0
  local mIoU = 0  -- mean IOU
  local mJacc = 0 -- Jacc

  -- write the results to output file if specified
  if opt.write_output then
      output_file = {version='VERSION 1.0', external_data={used='true', details='global_pool layer from BN-Inception pretrained from ActivityNet and ImageNet (https://github.com/yjxiong/anet2016-cuhk)'}}
      output_results = {}
  end

  while true do
    -- fetch a batch of data
    local data = loader_val:getBatch{batch_size = opt.batch_size, frames_per_video = opt.frames_per_video}
    local feats = torch.load(path.join(data.infos[1].file_path,'resnet_34_feat_mscoco.dat'))

    -- sample segments
    -- proposals is a table: {score, boundary, segments}
    -- score: k*L tensor     boundary: 2*k*L tensor
    -- segments: S*2 tensor (S is unknown)
    local proposals = protos.lm:sample(feats,{beam_size = opt.beam_size})  
    local prop_score = {proposals[1],proposals[2]}
    local clip_prop = proposals[3]

    -- evaluation
    local IoU_clip = 0
    local Jacc_clip = 0
    for i=1,data.clip_num do
        local best_jacc = 0
        local best_iou = 0
        for j=1,clip_prop:size(1) do
            local cur_jacc = jacc(data.segments[i], clip_prop[j])
            if cur_jacc > best_jacc then
                best_jacc = cur_jacc
            end
            local cur_iou = iou(data.segments[i], clip_prop[j])
            if cur_iou > best_iou then
               best_iou = cur_iou
            end
        end
        Jacc_clip = Jacc_clip + best_jacc
        IoU_clip = IoU_clip + best_iou
    end

    mJacc = mJacc + Jacc_clip/data.clip_num
    mIoU = mIoU + IoU_clip/data.clip_num
    -- counter
    n = n + 1

    -- visualization
    if opt.vis then visSegment(data.segments, clip_prop, data.infos[1].file_path) end

    -- write output to .json file
    if opt.write_output then
        local vid_id = string.sub(data.infos[1].file_path,-16,-6)
        local dur = loader_val.dur_frame_dict[vid_id][1]
        local tf = loader_val.dur_frame_dict[vid_id][2]
        local sampling_itv = math.ceil(tf/opt.frames_per_video)
        local time_per_sampled_frame = sampling_itv*dur/tf
        proposal_tmp = {}
        for i=1,clip_prop:size(1) do
            table.insert(proposal_tmp, {segment={(clip_prop[i][1]-1)*time_per_sampled_frame, (clip_prop[i][2]-1)*time_per_sampled_frame}, score=clip_prop[i][3]})
        end
        output_results[vid_id] = proposal_tmp
    end

    local ix0 = n
    local ix1 = val_images_use

    if verbose then
      print(string.format('evaluating validation performance... %d/%d (%f)', ix0, ix1, Jacc_clip/data.clip_num))
    end

    if n % 10 == 0 then collectgarbage() end
    if data.bounds.wrapped then break end -- the split ran out of data, lets break out
    if n >= val_images_use and val_images_use >= 0 then break end -- we've used enough images
  end

  -- average
  mIoU = mIoU/n
  mJacc = mJacc/n

  if opt.write_output then
      output_file['results'] = output_results
      utils.write_json('results-'..opt.id..'.json', output_file) -- write proposal results to a .json file
  end

  return mIoU, mJacc
end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0

local function lossFun()
  protos.lm:training()
  grad_params:zero()

  -- fetch a batch of data
  local data = loader:getBatch{batch_size = opt.batch_size, frames_per_video = opt.frames_per_video}
  local feats = torch.load(path.join(data.infos[1].file_path,'resnet_34_feat_mscoco.dat'))
  feats:add(torch.randn(feats:size()):type(feats:type())*0.001) -- add noise to the input features

  assert(feats:size()[1]/opt.frames_per_video == opt.batch_size)

  print(data.infos[1].file_path)
  -- forward pass
  -- prop_score is a four-tuple table: {score, boundary, gt_seg_enc, seg_prob}
  -- score: k*L tensor     boundary: 2*k*L tensor
  -- gt_seg_enc: S tensor  seg_prob: S*(mp_score_flatten_size+1)
  local prop_score = protos.lm:forward({feats,data.segments})  -- clip proposal scores  K*L
  local loss = protos.crit:forward(prop_score, data.segments)

  local dscore = protos.crit:backward(prop_score, data.segments)
  local dfeats = protos.lm:backward({feats,data.segments}, dscore)

  if opt.lm_weight_decay > 0 then
      -- we don't bother adding the l2 loss to the total loss
      grad_params:add(opt.lm_weight_decay, params)
      grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  local losses = { total_loss = loss}
  return losses
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local loss_history = {}
local val_miou_history = {}
local val_jacc_history = {}
local best_score

while true do
  -- eval loss/gradient
  local losses = lossFun()
  if iter % opt.losses_log_every == 0 then loss_history[iter] = losses.total_loss end
  print(string.format('iter %d: %f', iter, losses.total_loss))

  -- evaluation
  if (iter % opt.save_checkpoint_every == 0 or iter == opt.max_iters-1) then
      local val_miou, val_jacc = eval_split({val_images_use = opt.val_images_use})
      print('validation mIoU: ', val_miou, ' and validation Jacc: ', val_jacc)
      val_miou_history[iter] = val_miou
      val_jacc_history[iter] = val_jacc

      local checkpoint_path = path.join(opt.checkpoint_path, 'model_id' .. opt.id)

      -- write a json report and a t7 checkpoint
      local checkpoint = {}
      checkpoint.opt = opt
      checkpoint.iter = iter
      checkpoint.loss_history = loss_history
      checkpoint.val_miou_history = val_miou_history
      checkpoint.val_jacc_history = val_jacc_history

      utils.write_json(checkpoint_path .. '.json', checkpoint)
      print('wrote json checkpoint to ' .. checkpoint_path .. '.json')

      local current_jacc = val_jacc

      if best_score == nil or current_jacc >= best_score then
          best_score = current_jacc
          if iter > 0 then
              local save_protos = {}
              save_protos.lm = thin_lm:clearState()
              checkpoint.protos = save_protos
              torch.save(checkpoint_path .. '.t7', checkpoint)
              print('wrote checkpoint to ' .. checkpoint_path .. '.t7')
          end
      end
  end

  -- decay the learning rate for ProcNets
  local learning_rate = opt.learning_rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
    local decay_factor = math.pow(0.5, frac)
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  -- perform a parameter update
  if opt.optim == 'rmsprop' then
    rmsprop(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'adagrad' then
    adagrad(params, grad_params, learning_rate, opt.optim_epsilon, optim_state)
  elseif opt.optim == 'sgd' then
    sgd(params, grad_params, opt.learning_rate)
  elseif opt.optim == 'sgdm' then
    sgdm(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'sgdmom' then
    sgdmom(params, grad_params, learning_rate, opt.optim_alpha, optim_state)
  elseif opt.optim == 'adam' then
    adam(params, grad_params, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
  else
    error('bad option opt.optim')
  end

  -- stopping criterions
  iter = iter + 1
  if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
  -- if loss0 == nil then loss0 = losses.total_loss end
  -- if losses.total_loss > loss0 * 20 then
  --    print('loss seems to be exploding, quitting.')
  --    break
  -- end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
end
