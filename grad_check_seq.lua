require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'

require 'misc.ProcNets_bilstm_seq_guide'
local gradcheck = require 'misc.gradcheck'

local tests = {}
local tester = torch.Tester()

-- cutorch.manualSeed(123)
-- cutorch.setDevice(1)


-------------------------------------------------------------------------------
-- gradient check for the ts model
-------------------------------------------------------------------------------
local function gradCheckTS()

  local dtype = 'torch.DoubleTensor'
  local lmOpt = {}
  lmOpt.input_encoding_size = 8
  lmOpt.rnn_size = 9
  lmOpt.dropout = 0
  lmOpt.frames_per_video = 20 -- number of unrolled LSTMs
  lmOpt.batch_size = 1
  lmOpt.KTL = 3
  lmOpt.KTU = 9
  lmOpt.KTS = 2
  lmOpt.CN = 10

  local lm = nn.ProcNets(lmOpt)
  local crit = nn.ProcNetsCriterion({lmOpt.KTL, lmOpt.KTU, lmOpt.KTS})
  lm:type(dtype)
  crit:type(dtype)
  local ts = lm.ts
  -- imgs = torch.randn(lmOpt.KTL, lmOpt.frames_per_video):type(dtype)
  -- imgs = torch.Tensor(imgs_tmp:size(2),imgs_tmp:size(1)):copy(imgs_tmp:transpose(1,2))
  -- imgs = torch.randn(lmOpt.CN,2):type(dtype)
  imgs = torch.randn(lmOpt.frames_per_video,lmOpt.rnn_size):type(dtype)

  -- evaluate the analytic gradient
  local output = ts:forward(imgs)
  local w = torch.randn(output[1]:size())
  local w2 = torch.randn(output[2]:size())
  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output[1], w))+torch.sum(torch.cmul(output[2], w2))
  local gradOutput = {w, w2}
  local gradInput = ts:backward(imgs, gradOutput)

  -- create a loss function wrapper
  local function f(x)
    local output = ts:forward(x)
    local loss = torch.sum(torch.cmul(output[1], w))+torch.sum(torch.cmul(output[2], w2))
    return loss
  end
  local gradInput_num = gradcheck.numeric_gradient(f, imgs, 1, 1e-6)

  --[[
  print(gradInput)
  print(gradInput_num)
  local g = gradInput:view(-1)
  local gn = gradInput_num:view(-1)
  for i=1,g:nElement() do
    local r = gradcheck.relative_error(g[i],gn[i])
    print(i, g[i], gn[i], r)
  end
  ]]--
  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)

end


-------------------------------------------------------------------------------
-- gradient check for the language model
-------------------------------------------------------------------------------
-- test just the language model alone (without the criterion)
local function gradCheckLM()

  local dtype = 'torch.DoubleTensor'
  local lmOpt = {}
  lmOpt.input_encoding_size = 8
  lmOpt.rnn_size = 15
  lmOpt.dropout = 0
  lmOpt.frames_per_video = 20 -- number of unrolled LSTMs
  lmOpt.batch_size = 1
  lmOpt.KTL = 3
  lmOpt.KTU = 9
  lmOpt.KTS = 2
  lmOpt.CN = 5
  lmOpt.mp_scale_h = 4
  lmOpt.mp_scale_w = 4
  lmOpt.gradcheck = true

  local lm = nn.ProcNets(lmOpt)
  local crit = nn.ProcNetsCriterion({lmOpt.KTL, lmOpt.KTU, lmOpt.KTS})
  lm:type(dtype)
  crit:type(dtype)

  --[[
  local seq = torch.LongTensor(lmOpt.seq_length, lmOpt.batch_size):random(lmOpt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(lmOpt.batch_size, lmOpt.input_encoding_size):type(dtype)
  ]]--

  imgs = torch.randn(lmOpt.frames_per_video,lmOpt.input_encoding_size):type(dtype)
  segs = {{1,3},{5,8},{10,11},{13,17},{19,20}}

  -- evaluate the analytic gradient
  local output = lm:forward({imgs,segs})
  local w = torch.randn(output[1]:size())
  local w2 = torch.randn(output[2]:size())
  local w3 = torch.randn(output[3]:size())
  local w4 = torch.randn(output[4]:size())

  -- generate random weighted sum criterion
  local loss = torch.sum(torch.cmul(output[1], w)) + torch.sum(torch.cmul(output[2], w2)) + torch.sum(torch.cmul(output[3], w3)) + torch.sum(torch.cmul(output[4], w4))
  local gradOutput = {w, w2, w3, w4}
  local gradInput_t = lm:backward({imgs,segs}, gradOutput)
  local gradInput = gradInput_t[1]
  -- create a loss function wrapper
  local function f(x)
    local output = lm:forward({x,segs})
    local loss = torch.sum(torch.cmul(output[1], w)) + torch.sum(torch.cmul(output[2], w2)) + torch.sum(torch.cmul(output[3], w3)) + torch.sum(torch.cmul(output[4], w4))
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, imgs, 1, 1e-6)
  --[[
  print(gradInput)
  print(gradInput_num)
  local g = gradInput:view(-1)
  local gn = gradInput_num:view(-1)
  for i=1,g:nElement() do
    local r = gradcheck.relative_error(g[i],gn[i])
    print(i, g[i], gn[i], r)
  end
  ]]--
  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)
end


-------------------------------------------------------------------------------
-- gradient check
-------------------------------------------------------------------------------
-- test just the language model alone (without the criterion)
local function gradCheck()

  local dtype = 'torch.DoubleTensor'
  local lmOpt = {}
  lmOpt.input_encoding_size = 8
  lmOpt.vocab_size = 6
  lmOpt.rnn_size = 15
  lmOpt.dropout = 0
  lmOpt.frames_per_video = 20 -- number of unrolled LSTMs
  lmOpt.batch_size = 1
  lmOpt.KTL = 3
  lmOpt.KTU = 9
  lmOpt.KTS = 2
  lmOpt.CN = 5
  lmOpt.mp_scale_h = 4
  lmOpt.mp_scale_w = 4
  lmOpt.gradcheck = true

  local lm = nn.ProcNets(lmOpt)
  local crit = nn.ProcNetsCriterion({lmOpt.KTL, lmOpt.KTU, lmOpt.KTS})
  lm:type(dtype)
  crit:type(dtype)

  --[[
  local seq = torch.LongTensor(lmOpt.seq_length, lmOpt.batch_size):random(lmOpt.vocab_size)
  seq[{ {4, 7}, 1 }] = 0
  seq[{ {5, 7}, 4 }] = 0
  local imgs = torch.randn(lmOpt.batch_size, lmOpt.input_encoding_size):type(dtype)
  ]]--

  imgs = torch.randn(lmOpt.frames_per_video,lmOpt.input_encoding_size):type(dtype)

  label = {{1,3},{5,8},{10,11},{13,17},{19,20}}

  -- evaluate the analytic gradient
  local output = lm:forward({imgs,label})
  local loss = crit:forward(output,label)
  local gradOutput = crit:backward(output,label)
  local gradInput_t = lm:backward({imgs,label}, gradOutput)
  local gradInput = gradInput_t[1]
  -- create a loss function wrapper
  local function f(x)
    local output = lm:forward({x,label})
    local loss = crit:forward(output,label)
    return loss
  end

  local gradInput_num = gradcheck.numeric_gradient(f, imgs, 1, 1e-6)
  --[[
  print(gradInput)
  print(gradInput_num)
  local g = gradInput:view(-1)
  local gn = gradInput_num:view(-1)
  for i=1,g:nElement() do
    local r = gradcheck.relative_error(g[i],gn[i])
    print(i, g[i], gn[i], r)
  end
  --]]
  tester:assertTensorEq(gradInput, gradInput_num, 1e-4)
  tester:assertlt(gradcheck.relative_error(gradInput, gradInput_num, 1e-8), 1e-4)
end


tests.gradCheckTS = gradCheckTS
tests.gradCheckLM = gradCheckLM
tests.gradCheck = gradCheck

tester:add(tests)
tester:run()
