require 'torch'
require 'nn'
require 'nngraph'
require 'csvigo'

local utils = require 'misc.utils'
require 'misc.DataLoaderFeatExtract'
local net_utils = require 'misc.net_utils'
require 'misc.optim_updates'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train and validate a Procedure Segmentation model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-image_folder', '/z/home/luozhou/YouCookII/raw_frames_all', 'root directory of raw images')
cmd:option('-train_data_folder', 'train_raw_frames', 'training frame folder')
cmd:option('-train_feat_folder', 'train_feat_frame', 'training frame feature folder')
cmd:option('-val_data_folder', 'test_raw_frames', 'val/testing frame folder')
cmd:option('-val_feat_folder', 'test_feat_frame', 'val/testing frame feature folder')

cmd:option('-images_use', -1, 'number of videos to extract feature from. -1 = all')

-- Model settings
cmd:option('-frames_per_video', 500, 'number of frames sampled in one video')
cmd:option('-cnn_model_t7','/z/home/luozhou/dataset/e2eglstm/tc-resnet34/model_ide2eglstm-resnet-lr5-decay3.t7','path to CNN model t7 file containing the weights. For now it does not support other models than model_ide2eglstm-resnet-lr5-decay3.t7')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
local loader_val
loader_val = DataLoaderFeatExtract{folder_path = opt.image_folder, split = opt.val_data_folder}

-------------------------------------------------------------------------------
-- Initialize the networks
-------------------------------------------------------------------------------
local protos = {}
local loaded_checkpoint = torch.load(opt.cnn_model_t7)
protos = loaded_checkpoint.protos

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

collectgarbage()
protos.cnn:evaluate()
loader_val:resetIterator()

local function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-------------------------------------------------------------------------------
-- Feature extraction
-------------------------------------------------------------------------------
n=0
while true do
  -- fetch a batch of data
  local data = loader_val:getBatch{batch_size = opt.batch_size, frames_per_video = opt.frames_per_video}
  local feats_all=torch.CudaTensor(opt.frames_per_video, opt.input_encoding_size)
  local deco = 20
  for i=1,#data.images do
    if file_exists(path.join(data.infos[i].file_path,'resnet_34_feat_mscoco.dat')) == false then -- for .dat
    --  if file_exists(path.join(data.infos[i].file_path,'resnet_34_feat_mscoco.csv')) == false then -- for .csv
      print('[INFO] processing '.. data.infos[i].file_path)
      data.images[i] = net_utils.prepro(data.images[i], false, opt.gpuid >= 0) -- preprocess in place, and don't augment
      -- The input pixel value is between 0 and 255, which can directly feed into our pre-trained model on MSCOCO. Remember to divide data.images[i] by 255.0 if taking pre-trained Resnet models from ImageNet... Legacy issue on vgg/resnet input format...

      -- forward pass
      for j=1,deco do
           local feats = protos.cnn:forward(data.images[i]:sub((j-1)*opt.frames_per_video/deco+1,j*opt.frames_per_video/deco,1,3,1,224,1,224))
          feats_all:sub((j-1)*opt.frames_per_video/deco+1,j*opt.frames_per_video/deco,1,opt.input_encoding_size):copy(feats)
      end

      -- save feature to file
      torch.save(path.join(data.infos[i].file_path,'resnet_34_feat_mscoco.dat'),feats_all)
      -- csvigo.save{path = path.join(data.infos[i].file_path,'resnet_34_feat_mscoco.csv'), data=torch.totable(feats_all)}
    else
      print('[INFO]' .. data.infos[i].file_path .. 'has already been processed!' )
    end
  end 
  n=n+1
  if n >= opt.images_use and opt.images_use > 0 then break end -- we've used enough images 
end
