require 'nn'

----------------------------------------------------
-- Temporal Segmentation core
----------------------------------------------------

local layer, parent = torch.class('nn.TemporalSegmentation', 'nn.Module')
function layer:__init(kernelInfo, inputInfo, outputInfo)
    parent.__init(self)
    -- ts related parameters
    self.kernel_low = kernelInfo[1]
    self.kernel_high = kernelInfo[2]
    self.kernel_interval = kernelInfo[3]
    self.input_encoding_size = inputInfo[1]
    self.temporal_length = inputInfo[2]
    self.clip_number = outputInfo

    -- check if the conv parameters are correct
    assert((self.kernel_high-self.kernel_low)%self.kernel_interval==0)
    self.kernel_number = (self.kernel_high-self.kernel_low)/self.kernel_interval+1

    -- for now, kernel width should be odd, kernel interval should be even
    assert(self.kernel_high%2 == 1)
    assert(self.kernel_interval%2 == 0)

    -- layers definition
    self.tc = nn.ConcatTable()  -- score generator
    -- tc layer for score generation
    for i=1, self.kernel_number do
        self.tc:add(nn.TemporalConvolution(self.input_encoding_size, 3, self.kernel_low+(i-1)*self.kernel_interval))
    end

    -- transfer functions for proposal scores and boundaries
    self.transfer = nn.Sigmoid()
    self.transfer2 = nn.Tanh()
    self.transfer3 = nn.Tanh()
end

function layer:parameters()
    local params = {}
    local grad_params = {}
    p1, g1 = self.tc:parameters()
    for k,v in pairs(p1) do table.insert(params, v) end
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    return params, grad_params
end

function layer:training()
    self.tc:training()
end

function layer:evaluate()
    self.tc:evaluate()
end

function layer:updateOutput(input)
    
    assert(input:size(1) == self.temporal_length)
    assert(input:size(2) == self.input_encoding_size)

    -- forward through the (1) temporal convolution layer
    self.tc_output = torch.Tensor(self.kernel_number, self.temporal_length,3):type(input:type()):zero() -- fill -10 for those bad segments
    self.resize_input = {}

    local tc_o = self.tc:forward(input)

    for i=1, self.kernel_number do
        local kernel_size = self.kernel_low+(i-1)*self.kernel_interval
        self.tc_output[i]:sub((kernel_size+1)/2, self.temporal_length-(kernel_size-1)/2,1,3):copy(tc_o[i])
    end
     
    -- never use resize together with sub!!!
    local soft_tmp = torch.Tensor(self.kernel_number, self.temporal_length):copy(self.tc_output:sub(1,-1,1,-1,1,1)):type(input:type())
    local len_tmp = torch.Tensor(self.kernel_number, self.temporal_length):copy(self.tc_output:sub(1,-1,1,-1,2,2)):type(input:type())
    local center_tmp = torch.Tensor(self.kernel_number, self.temporal_length):copy(self.tc_output:sub(1,-1,1,-1,3,3)):type(input:type())
    self.soft_output = self.transfer:forward(soft_tmp)
    self.tc_len_output = 0.1*self.transfer2:forward(len_tmp)
    self.tc_center_output = 0.1*self.transfer3:forward(center_tmp)

    self.tc_boundary = torch.Tensor(2,self.soft_output:size(1),self.soft_output:size(2)):type(self.tc_center_output:type())

    for row=1,self.tc_boundary:size(2) do
      for column=1,self.tc_boundary:size(3) do
        -- adding offsets to clip center and length
        local len = ((row-1)*self.kernel_interval+self.kernel_low)*torch.exp(self.tc_len_output[{row,column}])
        local center = column+self.tc_center_output[{row,column}]*((row-1)*self.kernel_interval+self.kernel_low)

        self.tc_boundary[{1,row,column}] = center-len/2  -- lower boundary
        self.tc_boundary[{2,row,column}] = center+len/2  -- upper boundary
      end
    end

    self.output = {self.soft_output, self.tc_boundary}
    return self.output
end

function layer:updateGradInput(input, gradOutput)
    assert(#gradOutput == 2) -- loss for score, length offset and center offset
    local dlen = (gradOutput[2]:sub(2,2,1,-1,1,-1) - gradOutput[2]:sub(1,1,1,-1,1,-1))/2
    local dcenter = gradOutput[2]:sub(2,2,1,-1,1,-1) + gradOutput[2]:sub(1,1,1,-1,1,-1)
    local dtc_len = torch.Tensor(self.soft_output:size()):type(self.soft_output:type()):zero()
    local dtc_center = torch.Tensor(self.soft_output:size()):type(self.soft_output:type()):zero()

    for row=1,gradOutput[2]:size(2) do
      for column=1,gradOutput[2]:size(3) do
        dtc_len[{row,column}] = dlen[{1,row,column}]*((row-1)*self.kernel_interval+self.kernel_low)*torch.exp(self.tc_len_output[{row,column}])
        dtc_center[{row,column}] = dcenter[{1,row,column}]*((row-1)*self.kernel_interval+self.kernel_low)
      end
    end

    local soft_tmp = torch.Tensor(self.kernel_number, self.temporal_length):copy(self.tc_output:sub(1,-1,1,-1,1,1)):type(input:type())
    local len_tmp = torch.Tensor(self.kernel_number, self.temporal_length):copy(self.tc_output:sub(1,-1,1,-1,2,2)):type(input:type())
    local center_tmp = torch.Tensor(self.kernel_number, self.temporal_length):copy(self.tc_output:sub(1,-1,1,-1,3,3)):type(input:type())

    local d_soft = self.transfer:backward(soft_tmp,gradOutput[1]) 
    dtc_len = 0.1*self.transfer2:backward(len_tmp,dtc_len)
    dtc_center = 0.1*self.transfer3:backward(center_tmp,dtc_center)
    local d_tc = torch.Tensor(self.tc_output:size()):type(self.tc_output:type())
    d_tc:sub(1,-1,1,-1,1,1):copy(d_soft)
    d_tc:sub(1,-1,1,-1,2,2):copy(dtc_len)
    d_tc:sub(1,-1,1,-1,3,3):copy(dtc_center)

    d_tc_temp = {}
    for i=1,d_tc:size(1) do      
        local kernel_size = self.kernel_low+(i-1)*self.kernel_interval
        -- (deprecated) need to normalize the gradients
        table.insert(d_tc_temp, torch.Tensor(d_tc[i]:size(1)-kernel_size+1,3):copy(d_tc[i]:sub((kernel_size+1)/2,self.temporal_length-(kernel_size-1)/2,1,3)):type(gradOutput[1]:type()))
    end
    self.gradInput = self.tc:backward(input,d_tc_temp)

    return self.gradInput
end

-----------------------------------------------------
-- Best Proposal Layer
-----------------------------------------------------
local layerBP, parent = torch.class('nn.BestProposal', 'nn.Module')
function layerBP:__init(top_number, base, interval, mp_scale)
    parent.__init(self)
    self.top_n = top_number
    self.base = base
    self.interval = interval
    self.mp_scale = mp_scale
end

-- forward only, no need for backprop in this work
function layerBP:updateOutput(input_table)
    local input = input_table[1]
    local boundary = input_table[2]
    local k_number = input:size(1)
    local t_len = input:size(2)
    local resize_input = torch.Tensor(input:size()):copy(input):type(input:type())
    resize_input:resize(1, k_number*t_len)
    local value, ind = torch.topk(resize_input, self.top_n, true, true)
    -- kernel_index and temporal_location
    local index = torch.Tensor(ind:size()):copy(ind):type(input:type())
    local ki = torch.floor((index-1)/t_len)+1 -- row
    local tl = index-t_len*(ki-1) -- column
    -- temp_output stores the clip location as well as the input location
    local temp_output = torch.Tensor(self.top_n,2):zero():type(input:type())
    for i=1, self.top_n do
        -- starting frame
        temp_output[{i,1}] = boundary[{1,ki[{1,i}],tl[{1,i}]}]
        -- ending frame
        temp_output[{i,2}] = boundary[{2,ki[{1,i}],tl[{1,i}]}]
        -- max and min boundaries
        temp_output[{i,1}] = math.max(1,temp_output[{i,1}])
        temp_output[{i,2}] = math.max(1,temp_output[{i,2}])
        temp_output[{i,1}] = math.min(t_len*self.mp_scale,temp_output[{i,1}])
        temp_output[{i,2}] = math.min(t_len*self.mp_scale,temp_output[{i,2}])       
    end  

    self.output:resize(self.top_n, 2)

    self.output:copy(temp_output)
    return self.output
end
