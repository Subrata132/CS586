import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, if_pool=True):
        super(Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.if_pool = if_pool

        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride
        )
        self.batch_normalization = nn.BatchNorm2d(num_features=self.out_channels)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.if_pool:
            x = self.pool(x)
        x = self.activation(x)
        x = self.batch_normalization(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, in_channel=2, num_class=5):
        super(CNNModel, self).__init__()
        self.conv_block_1 = Block(in_channels=in_channel, out_channels=64)
        self.conv_block_2 = Block(in_channels=64, out_channels=128)
        self.conv_block_3 = Block(in_channels=128, out_channels=256, if_pool=False)

        self.flatten = nn.Flatten()

        self.linear_1 = nn.Linear(in_features=2048, out_features=256)
        self.linear_2 = nn.Linear(in_features=256, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=num_class)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.out(x)
        return x


def main():
    x = torch.randn(16, 2, 18, 48)
    cnn_model = CNNModel()
    x = cnn_model(x)
    print(x.shape)


if __name__ == '__main__':
    main()
