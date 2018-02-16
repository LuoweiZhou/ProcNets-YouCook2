require 'nn'

local layer, parent = torch.class('nn.ScoreMaxpool', 'nn.Module')

function layer:__init(kernel_size_w,kernel_size_h, gradcheck)
  parent.__init(self)
  self.kernelsize_w = kernel_size_w
  self.kernelsize_h = kernel_size_h
  self.mp = nn.SpatialMaxPooling(kernel_size_w,kernel_size_h,kernel_size_w,kernel_size_h)
  self.gradcheck = gradcheck
end

function layer:updateOutput(input)
  local score = input[1] -- of dimension k*L
  local boundary = input[2] -- of dimension k*L*2
  local kernel_num = score:size(1)
  local frame_num = score:size(2)

  local maxpool_score = self.mp:forward(score:view(1,kernel_num,frame_num))
  local index = self.mp.indices
  local maxpool_boundary = torch.Tensor(2,kernel_num/self.kernelsize_h,frame_num/self.kernelsize_w):type(score:type())
  -- manipulate boundary cells according to max pooling index
  for i=1,kernel_num/self.kernelsize_h do
    for j=1,frame_num/self.kernelsize_w do
      if self.gradcheck then
        -- some index issue here while gradient checking...
        maxpool_boundary[{1,i,j}] = boundary:storage()[index[{1,i,j}]]
        maxpool_boundary[{2,i,j}] = boundary:storage()[index[{1,i,j}]+kernel_num*frame_num] 
      else
        maxpool_boundary[{1,i,j}] = boundary:storage()[index[{1,1,i,j}]]
        maxpool_boundary[{2,i,j}] = boundary:storage()[index[{1,1,i,j}]+kernel_num*frame_num]
      end
    end
  end
  self.output = {maxpool_score:view(maxpool_score:size(2),maxpool_score:size(3)), maxpool_boundary}
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local dscore = gradOutput[1]
  local dboundary = gradOutput[2]
  local index = self.mp.indices
  local kernel_num = input[1]:size(1)
  local frame_num = input[1]:size(2)

  local dinput_s = self.mp:backward(input[1]:view(1,kernel_num,frame_num),dscore:view(1,dscore:size(1),dscore:size(2)))
  local dinput_b = torch.Tensor(input[2]:size()):type(input[2]:type()):zero()
  -- manipulate boundary gradient cells according to max pooling index  
  -- for i=1,kernel_num/self.kernelsize_h do
  --   for j=1,frame_num/self.kernelsize_w do
  --     dinput_b:storage()[index[{1,1,i,j}]] = dboundary[{1,i,j}]
  --     dinput_b:storage()[index[{1,1,i,j}]+kernel_num*frame_num] = dboundary[{2,i,j}]
  --   end
  -- end
  self.gradInput = {dinput_s:view(dinput_s:size(2),dinput_s:size(3)), dinput_b}
  return self.gradInput
end
