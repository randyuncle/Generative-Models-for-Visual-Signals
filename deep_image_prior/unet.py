import torch.nn as nn
import torch as torch

# The convolution layer built from the source code
def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)

    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)

###
# The encoder-decoder structure similar to the original paper
# The following architecture refers to the following github repo:
# https://github.com/RPraneetha/DL-RP41-Deep-Image-Prior/tree/master
###

class DownSample(nn.Module):
    def __init__(self, in_size, out_size, kernel, pad, downsample_mode):
        super(DownSample, self).__init__()

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_size)
        self.cnn1 = conv(in_size, out_size, kernel, 1, True, pad, downsample_mode)
        self.cnn2 = conv(out_size, out_size, kernel, 1, True, pad, downsample_mode)
    
    def forward(self, input):
        output = self.cnn1(input)
        output = self.batch_norm(output)
        output = self.act(output)
        
        output = self.cnn2(output)
        output = self.batch_norm(output)
        output = self.act(output)
        
        return output

class UpSample(nn.Module):
    def __init__(self, in_size, out_size, kernel, pad, upsample_mode):
        super(UpSample, self).__init__()
        
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_size)
        self.batch_norm_fixed = nn.BatchNorm2d(in_size)
        self.cnn1 = conv(in_size, out_size, kernel, 2, True, pad, 'stride')
        self.cnn2 = conv(out_size, out_size, 1, 1, True, pad, 'stride')
        self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode)

    def forward(self, input):
        output = self.batch_norm_fixed(input)
        
        output = self.cnn1(output)
        output = self.batch_norm(output)
        output = self.act(output)
        
        output = self.cnn2(output)
        output = self.batch_norm(output)
        output = self.act(output)
        
        output = self.upsample(output)
        
        return output

class SkipConnection(nn.Module):
    def __init__(self, in_size, out_size, kernel, pad):
        super(SkipConnection, self).__init__()
        
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.batch_norm = nn.BatchNorm2d(out_size)
        self.cnn = conv(in_size, out_size, kernel, 1, True, pad)

    def forward(self, input):
        output = self.cnn(input)
        output = self.batch_norm(output)
        output = self.act(output)
        
        return output
        
class SkipArchitecture(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, 
                num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
                filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                need_sigmoid=True, need_bias=True, 
                pad='zero', upsample_mode='nearest', downsample_mode='stride'):
        super(SkipArchitecture, self).__init__()
        
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
        self.n_scales = len(num_channels_down)
        
        self.down_sampler = nn.ModuleList([
            DownSample(
                in_size=num_input_channels if i == 0 else num_channels_down[i-1], 
                out_size=num_channels_down[i], kernel=filter_size_down, pad=pad, 
                downsample_mode=downsample_mode
            ) for i in range(len(num_channels_down))
        ])
        self.up_sampler = nn.ModuleList([
            UpSample(
                in_size=num_channels_skip[i] + num_channels_up[i + 1] if i != len(num_channels_down) - 1
                        else num_channels_up[i],
                out_size=num_channels_up[i], kernel=filter_size_up, pad=pad, 
                upsample_mode=upsample_mode
            ) for i in range(len(num_channels_up)-1, -1, -1)
        ])
        self.skip_connections = nn.ModuleList([
            SkipConnection(
                in_size=num_channels_down[i], out_size=num_channels_skip[i], 
                kernel=filter_skip_size, pad=pad
            ) for i in range(len(num_channels_up))
        ])
        
        self.last_cnn = conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)
        self.sigmoid = nn.Sigmoid() if need_sigmoid else None

    def forward(self, input):
        skip = []
        
        for i in range(self.n_scales):
            # print(self.down_sampler[i])
            input = self.down_sampler[i].forward(input)
            skip.append(self.skip_connections[i].forward(input))
            
        for i in range(self.n_scales):
            if i == 0:
                # print(input)
                # print(self.up_sampler[i])
                input = self.up_sampler[i].forward(skip[-1])
            else:
                input = self.up_sampler[i].forward(torch.cat([input, skip[self.n_scales - i - 1]], 1))

        output = self.last_cnn(input)
        if self.sigmoid != None:
            output = self.sigmoid(output)
        
        return output