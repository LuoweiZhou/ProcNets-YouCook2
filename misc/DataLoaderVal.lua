-- Read binary feature files from image folders (val set)

local utils = require 'misc.utils'
require 'lfs'
require 'image'
require 'csvigo'

local DataLoaderVal = torch.class('DataLoaderVal')

function DataLoaderVal:__init(opt)

  -- load the json file which contains additional information about the dataset
  local split_path = path.join(opt.folder_path, opt.split)
  print('DataLoaderVal loading videos from folder: ', split_path)

  -- load the val video duration & total frame info
  self.dur_frame_dict = {}
  dur_frame_table = csvigo.load({path = opt.val_vidinfo_file, mode = 'large'})
  for i=2, #dur_frame_table do
      self.dur_frame_dict[dur_frame_table[i][1]] = {dur_frame_table[i][2], dur_frame_table[i][3]}
  end

  -- load the dataset .json file
  self.ann_info = utils.read_json(opt.ann_file).database
 
  local function file_exists(name)
      local f=io.open(name,"r")
      if f~=nil then io.close(f) return true else return false end
  end

  self.folders = {}
  self.ids = {}
  self.vid_id = {}
  self.aug = {}
  self.shuffled_folders = {}
  self.shuffled_vid_id = {}
  self.shuffled_aug = {}
  self.iterator = 1

  print('listing all video2frame folders in directory ' .. split_path)
  local n = 1
  for file in paths.files(split_path) do
      if file ~= '.' and file ~= '..' then
          print('loading videos from catagory ' .. file .. '...')
          local categorypath = path.join(split_path, file)
          for videofolder in paths.files(categorypath) do 
              -- the temporal segments ground-truth must exist
              if videofolder ~='.' and videofolder ~= '..' and self.ann_info[videofolder] and file_exists(path.join(categorypath,videofolder,'0001','resnet_34_feat_mscoco.dat')) then
                  local clippath = path.join(categorypath, videofolder)
                  local aug_counter = 0
                      local clipfolder = "0001"
                      local fullpath = path.join(clippath, clipfolder)
                      table.insert(self.folders, fullpath)
                      table.insert(self.ids, n) -- just order them sequentially
                      table.insert(self.vid_id, videofolder)
                      table.insert(self.aug, tonumber(string.sub(clipfolder,-2,-1)))
                      n=n+1
              end
          end
      end
  end

  -- no need to shuffle for the validation set!
  self.shuffled_folders = self.folders
  self.shuffled_vid_id = self.vid_id
  self.shuffled_aug = self.aug
  
  self.N = #self.shuffled_folders
  print('DataLoaderVal found ' .. self.N .. 'video clips!')
end

function DataLoaderVal:resetIterator()
  self.iterator = 1
end

function DataLoaderVal:getBatch(opt)
  local batch_size = utils.getopt(opt, 'batch_size', 1) -- how many images get returned at one time (to go through CNN)
  -- pick an index of the datapoint to load next
  local frames_per_video = utils.getopt(opt, 'frames_per_video', 500)
  local tempo_seg = {}
  local max_index = self.N
  local wrapped = false
  local infos = {}

  local function file_exists(name)
      local f=io.open(name,"r")
      if f~=nil then io.close(f) return true else return false end
  end

  -- round to the nearest integer
  local function round(num, idp)
      local mult = 10^(idp or 0)
      return math.floor(num * mult + 0.5) / mult
  end

  -- batch_size is always one...
  for i=1,batch_size do
    local ri = self.iterator
    local ri_next = ri + 1 -- increment iterator
    if ri_next > max_index then ri_next = 1; wrapped = true end -- wrap back around
    self.iterator = ri_next

    local vid_id = self.shuffled_vid_id[ri]
    local duration_t = self.dur_frame_dict[vid_id][1]
    local total_frame = self.dur_frame_dict[vid_id][2]
    local sampling_itv = math.ceil(total_frame/frames_per_video)
    local time_per_sampled_frame = sampling_itv*duration_t/total_frame

    ann = self.ann_info[vid_id].annotations
    for j=1,#ann do
        local start_t = ann[j].segment[1]
        local end_t = ann[j].segment[2]
        local start_f = start_t/time_per_sampled_frame+1
        local end_f = end_t/time_per_sampled_frame+1
        tempo_seg[j] = {start_f, end_f}
    end
    
    -- and record associated info as well
    local info_struct = {}
    info_struct.id = self.ids[ri]
    info_struct.file_path = self.shuffled_folders[ri]
    table.insert(infos, info_struct)
  end

  local data = {}
  data.segments = tempo_seg
  data.clip_num = #tempo_seg
  data.bounds = {it_pos_now = self.iterator, it_max = self.N, wrapped = wrapped}
  data.infos = infos
  return data
end
