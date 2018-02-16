-- Dataloader for generating feature files

local utils = require 'misc.utils'
require 'lfs'
require 'image'

local DataLoaderFeatExtract = torch.class('DataLoaderFeatExtract')

function DataLoaderFeatExtract:__init(opt)

  -- load the json file which contains additional information about the dataset
  local split_path = path.join(opt.folder_path, opt.split)
  print('DataLoaderFeatExtract loading videos from folder: ', split_path)

  self.folders = {}
  self.ids = {}
  self.video_flag = {} -- decide which clips belong to a video
  self.shuffled_folders = {}
  self.iterator = 1

  print('listing all video2frame folders in directory ' .. split_path)
  local n = 1
  for file in paths.files(split_path) do
      if file ~= '.' and file ~= '..' then
          print('loading videos from catagory ' .. file .. '...')
          local categorypath = path.join(split_path, file)
          for videofolder in paths.files(categorypath) do 
              if videofolder ~='.' and videofolder ~= '..' then
                  local clippath = path.join(categorypath, videofolder)   
                  for clipfolder in paths.files(clippath) do
                      if clipfolder ~='.' and clipfolder ~= '..' then
                          local fullpath = path.join(clippath, clipfolder)
                          table.insert(self.folders, fullpath)
                          table.insert(self.ids, n) -- just order them sequentially
                          table.insert(self.video_flag, 0)
                          n=n+1
                      end
                  end
                  self.video_flag[#self.video_flag]=1
              end
          end
      end
  end

  -- no need for shuffle
  self.shuffled_folders = self.folders
  
  self.N = #self.shuffled_folders
  print('DataLoaderFeatExtract found ' .. self.N .. 'video clips!')
end

function DataLoaderFeatExtract:resetIterator()
  self.iterator = 1
end

function DataLoaderFeatExtract:getBatch(opt)
  -- local batch_size = utils.getopt(opt, 'batch_size', 1) -- how many images get returned at one time (to go through CNN)
  -- pick an index of the datapoint to load next
  local frames_per_video = utils.getopt(opt, 'frames_per_video', 500)
  local img_batch_raw = {}
  local clip_counter = 0
  local max_index = self.N
  local wrapped = false
  local infos = {}

  local function file_exists(name)
      local f=io.open(name,"r")
      if f~=nil then io.close(f) return true else return false end
  end

  while true do
    local ri = self.iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterator = ri_next

    clip_counter = clip_counter+1
    video_f = self.video_flag[ri]
    -- load the image
    local j = 1   
    local videoframe
    img_batch_raw[clip_counter] = torch.ByteTensor(frames_per_video, 3, 256, 256)
    for fn=1, frames_per_video do
        assert(frames_per_video<1000) -- cannot have more than 999 frames per video
        videoframe = string.format('%.4d.jpg',fn)
       
        if file_exists(path.join(self.shuffled_folders[ri], videoframe)) then
            local img = image.load(path.join(self.shuffled_folders[ri], videoframe), 3, 'byte')
            img_batch_raw[clip_counter][j] = image.scale(img, 256, 256)
            j=j+1
        end
    end
    if j ~= frames_per_video+1 then
        print('not enough frames!' .. '---'.. self.shuffled_folders[ri])
    end
    -- and record associated info as well
    local info_struct = {}
    info_struct.id = self.ids[ri]
    info_struct.file_path = self.shuffled_folders[ri]
    table.insert(infos, info_struct)
    if self.video_flag[ri]==1 then
        break
    end
  end

  local data = {}
  data.images = img_batch_raw
  data.bounds = {it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}
  data.infos = infos
  data.aug = clip_counter
  return data
end
