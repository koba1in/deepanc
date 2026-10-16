import torch
import torch.nn

class ConvBlock(torch.nn.Module):
    
    def __init__(self, prev_channels, channels, conv_kernels, conv_strides, conv_padding, alpha):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(prev_channels, channels, conv_kernels, conv_strides, conv_padding, ),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ELU(alpha)
        )
    
    def forward(self, x):
        # identity = x
        x = self.conv(x)
        return x
    
class DecBlock(torch.nn.Module):
    
    def __init__(self, prev_channels, channels, conv_kernels, conv_strides, conv_padding, output_padding, alpha, activate=True):
        super().__init__()
        if activate:
            self.dec = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(prev_channels, channels, conv_kernels, conv_strides, conv_padding, output_padding),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ELU(alpha)
            )
        else:
            self.dec = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(prev_channels, channels, conv_kernels, conv_strides, conv_padding, output_padding),
                # torch.nn.BatchNorm2d(channels),
            )
        
    def forward(self, id, x):
        x = torch.concat((id, x), dim=1)
        x = self.dec(x)
        return x
    
# class RNNBlock(torch.nn.Module):
    
#     def __init__(self, input_size, hidden_size, num_layers, alpha):
#         super().__init__()
#         self.rnn = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         # self.elu = torch.nn.ELU(alpha)
    
#     def forward(self, x):
#         x, _ = self.rnn(x)
#         # x = self.elu(x)
#         return x
    
class CRN(torch.nn.Module):
    
    def __init__(self, channels, ):
        # [2, T, 320]
        super().__init__()
        
        prev_channel = 2
        conv_kernel = (1, 3)
        conv_stride = (1, 2)
        conv_padding = (0, 0)
        output_padding = (0, 0)
        alpha = 1
        self.ConvBlocks = torch.nn.ModuleList()
        for channel in channels:
            self.ConvBlocks.append(ConvBlock(prev_channel, channel, conv_kernel, conv_stride, conv_padding, alpha ))
            prev_channel = channel
        
        input_size = 1024
        hidden_size = 1024
        num_layers = 2
        self.RNNBlocks = RNNBlock(input_size, hidden_size, num_layers, alpha)
        self.DecBlocksReal = torch.nn.ModuleList()
        self.DecBlocksImage = torch.nn.ModuleList()
        for i in range(-2, -len(channels), -1):
            channel = channels[i]
            self.DecBlocksReal.append(DecBlock(prev_channel*2, channel, conv_kernel, conv_stride, conv_padding, output_padding, alpha))
            self.DecBlocksImage.append(DecBlock(prev_channel*2, channel, conv_kernel, conv_stride, conv_padding, output_padding, alpha))
            prev_channel = channel
        channel = channels[0]
        output_padding = (0, 1)
        self.DecBlocksReal.append(DecBlock(prev_channel*2, channel, conv_kernel, conv_stride, conv_padding, output_padding, alpha))
        self.DecBlocksImage.append(DecBlock(prev_channel*2, channel, conv_kernel, conv_stride, conv_padding, output_padding, alpha))
        prev_channel = channel
        output_padding = (0, 0)
        self.DecBlocksReal.append(DecBlock(prev_channel*2, 1, conv_kernel, conv_stride, conv_padding, output_padding, alpha, activate=False))
        self.DecBlocksImage.append(DecBlock(prev_channel*2, 1, conv_kernel, conv_stride, conv_padding, output_padding, alpha, activate=False))  

    def forward(self, x):
        #x: [Batch, channel, T, 161]
        res = []
        for ConvBlock in self.ConvBlocks:
            x = ConvBlock(x)
            res.append(x)
        x = torch.permute(x, (0, 2, 1, 3))
        x = x.reshape(x.size(0), x.size(1), -1)
        #x: [Batch, T, 1024]
        x = self.RNNBlocks(x)
        x = x.reshape(x.size(0), x.size(1), 256, 4)
        x = torch.permute(x, (0, 2, 1, 3))
        #x: [Batch, channel, T, 4]
        x_real = None
        x_image = None
        for DecBlockReal, DecBlockImage in zip(self.DecBlocksReal, self.DecBlocksImage):
            id = res.pop()
            if x_real is None:
                x_real = DecBlockReal(id, x)
                x_image = DecBlockImage(id, x)
            else:
                x_real = DecBlockReal(id, x_real)
                x_image = DecBlockImage(id, x_image)
        x = torch.concat((x_real, x_image), dim=1)
        #x: [Batch, 2, T, 161]
        return x

class RNNBlock(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, alpha=None, groups=2):
        super().__init__()

        if input_size % groups != 0 or hidden_size % groups != 0:
            raise ValueError("input_size and hidden_size must be divisible by groups")

        self.groups = groups
        self.layers = torch.nn.ModuleList()

        for layer_index in range(num_layers):
            layer_input_size = input_size if layer_index == 0 else hidden_size
            group_lstms = torch.nn.ModuleList([
                torch.nn.LSTM(
                    input_size=layer_input_size // groups,
                    hidden_size=hidden_size // groups,
                    batch_first=True,
                )
                for _ in range(groups)
            ])
            self.layers.append(group_lstms)

    def forward(self, x):
        # x: [batch, time, features]
        for layer_index, group_lstms in enumerate(self.layers):
            group_inputs = torch.chunk(x, self.groups, dim=-1)
            group_outputs = [
                lstm(group_input)[0]
                for lstm, group_input in zip(group_lstms, group_inputs)
            ]
            x = torch.cat(group_outputs, dim=-1)

            # 2層目に進む前に、群ごとの特徴を交互に並べ替える
            if layer_index < len(self.layers) - 1:
                batch, time, features = x.shape
                x = (
                    x.reshape(batch, time, self.groups, features // self.groups)
                     .transpose(2, 3)
                     .contiguous()
                     .reshape(batch, time, features)
                )

        return x

crn_model = CRN(channels=[16, 32, 64, 128, 256])

