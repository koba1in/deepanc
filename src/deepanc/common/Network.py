import torch
import torch.nn

class ConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, alpha=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ELU(alpha)
        )
    
    def forward(self, x):
        # identity = x
        x = self.conv(x)
        return x

class DecBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, alpha=1, activate=True):
        super().__init__()

        self.conv_transpose = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activate = activate
        if activate:
            self.activate = torch.nn.Sequential(
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ELU(alpha)
            )

    def forward(self, id, x, output_size):
        x = torch.concat((id, x), dim=1)
        x = self.conv_transpose(x, output_size=output_size)
        if self.activate:
            x = self.activate(x)
        return x

class RNNBlock(torch.nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, groups=2):
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

class CRN(torch.nn.Module):
    
    def __init__(self, channels, ):
        # [2, T, 320]
        super().__init__()
        
        prev_channel = 2
        kernel_size = (1, 3)
        stride = (1, 2)
        padding = (0, 0)
        alpha = 1

        ConvBlocks = []
        DecBlocksReal = []
        DecBlocksImage = []

        for channel in channels:
            ConvBlocks.append(ConvBlock(prev_channel, channel, kernel_size, stride, padding, alpha))

            if prev_channel == 2:
                DecBlocksReal.append(DecBlock(2*channel, 1, kernel_size, stride, padding, activate=False))
                DecBlocksImage.append(DecBlock(2*channel, 1, kernel_size, stride, padding, activate=False))
            else:
                DecBlocksReal.append(DecBlock(2*channel, prev_channel, kernel_size, stride, padding, alpha))
                DecBlocksImage.append(DecBlock(2*channel, prev_channel, kernel_size, stride, padding, alpha))

            prev_channel = channel

        self.ConvBlocks = torch.nn.ModuleList(ConvBlocks)

        self.DecBlocksReal = torch.nn.ModuleList(reversed(DecBlocksReal))
        self.DecBlocksImage = torch.nn.ModuleList(reversed(DecBlocksImage))

        self.RNNBlocks = RNNBlock(input_size=1024, hidden_size=1024, num_layers=2,)

    def forward(self, x):
        #x: [Batch, channel, T, 161]
        res = []
        sizes = []
        sizes.append((x.size(0), 1, x.size(2), x.size(3)))
        
        for ConvBlock in self.ConvBlocks:
            x = ConvBlock(x)
            res.append(x)
            sizes.append(x.size())
        sizes.pop()
        
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
            size = sizes.pop()
            if x_real is None:
                x_real = DecBlockReal(id, x, size)
                x_image = DecBlockImage(id, x, size)
            else:
                x_real = DecBlockReal(id, x_real, size)
                x_image = DecBlockImage(id, x_image, size)
        x = torch.concat((x_real, x_image), dim=1)
        #x: [Batch, 2, T, 161]
        return x



crn_model = CRN(channels=[16, 32, 64, 128, 256])
