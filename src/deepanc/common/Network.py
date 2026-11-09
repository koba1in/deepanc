import torch
import torch.nn
from enum import StrEnum, auto

class ModelType(StrEnum):
    NONE = auto()
    SKIP = auto()
    RES = auto()

class ConvBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, alpha=1):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ELU(alpha)
        )
    
    def forward(self, x):
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
        if id is not None:
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
    
    def __init__(self, channels, modeltype: ModelType):
        super().__init__()
        if not isinstance(modeltype, ModelType):
            raise TypeError(f"modeltype must be a ModelType, got {type(modeltype).__name__}")
        self.modeltype = modeltype

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
            match modeltype:
                case ModelType.SKIP:
                    if prev_channel == 2:
                        DecBlocksReal.append(DecBlock(2*channel, 1, kernel_size, stride, padding, activate=False))
                        DecBlocksImage.append(DecBlock(2*channel, 1, kernel_size, stride, padding, activate=False))
                    else:
                        DecBlocksReal.append(DecBlock(2*channel, prev_channel, kernel_size, stride, padding, alpha))
                        DecBlocksImage.append(DecBlock(2*channel, prev_channel, kernel_size, stride, padding, alpha))
                case _:
                    if prev_channel == 2:
                        DecBlocksReal.append(DecBlock(channel, 1, kernel_size, stride, padding, activate=False))
                        DecBlocksImage.append(DecBlock(channel, 1, kernel_size, stride, padding, activate=False))
                    else:
                        DecBlocksReal.append(DecBlock(channel, prev_channel, kernel_size, stride, padding, alpha))
                        DecBlocksImage.append(DecBlock(channel, prev_channel, kernel_size, stride, padding, alpha))

            prev_channel = channel

        self.ConvBlocks = torch.nn.ModuleList(ConvBlocks)

        self.DecBlocksReal = torch.nn.ModuleList(reversed(DecBlocksReal))
        self.DecBlocksImage = torch.nn.ModuleList(reversed(DecBlocksImage))

        self.RNNBlocks = RNNBlock(input_size=1024, hidden_size=1024, num_layers=2,)

    def forward(self, x):
        #x: [Batch, channel, T, freq]

        features = []
        sizes = []
        sizes.append((x.size(0), 1, x.size(2), x.size(3)))

        # Conv
        for ConvBlock in self.ConvBlocks:
            x = ConvBlock(x)

            features.append(x)
            sizes.append(x.size())

        sizes.pop()
        # [Batch, channel, T, freq]

        # RNN
        x = torch.permute(x, (0, 2, 1, 3))
        temp_size = x.size()
        x = x.reshape(x.size(0), x.size(1), -1)
        # [Batch, T, channel * freq]
        x = self.RNNBlocks(x)
        x = x.reshape(temp_size)
        x = torch.permute(x, (0, 2, 1, 3))
        # [Batch, channel, T, freq]
        
        # DecConv
        x_real = None
        x_image = None
        skip_feature = None
        for DecBlockReal, DecBlockImage in zip(self.DecBlocksReal, self.DecBlocksImage):
            match self.modeltype:
                case ModelType.NONE:
                    pass
                case ModelType.SKIP:
                    skip_feature = features.pop()
                case ModelType.RES:
                    res = features.pop()
                    if x_real is None:
                        x = x + res
                    else:
                        x_real, x_image = x_real + res, x_image + res
            size = sizes.pop()
            if x_real is None:
                x_real = DecBlockReal(skip_feature, x, size)
                x_image = DecBlockImage(skip_feature, x, size)
            else:
                x_real = DecBlockReal(skip_feature, x_real, size)
                x_image = DecBlockImage(skip_feature, x_image, size)
        x = torch.concat((x_real, x_image), dim=1)
        #x: [Batch, 2, T, freq]
        return x


default_channels = [16, 32, 64, 128, 256]

# model names
# crn_model = CRN(default_channels, ModelType.SKIP)
# crn_noskipconnection_model = CRN(default_channels, ModelType.NONE)
# crn_res_model = CRN(default_channels, ModelType.RES)
