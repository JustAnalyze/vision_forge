
'''
Script for building models
'''

import torch
import torch.nn as nn

# create our model architecture
class TinyVGG(nn.Module):
  def __init__(self, in_filters: int, out_filters: int, output_shape: int):
    super().__init__()

    self.conv_blk_1 = nn.Sequential(
        nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=(3,3), stride=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=(3,3), stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2))

    self.conv_blk_2 = nn.Sequential(
        nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=(3,3), stride=1),
        nn.ReLU(),

        nn.Conv2d(in_channels=out_filters, out_channels=out_filters, kernel_size=(3,3), stride=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2,2), stride=2))

    self.fc = nn.Sequential(
        nn.Flatten(1, -1),
        nn.Linear(in_features=13*13*out_filters, out_features=output_shape))

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    output = self.conv_blk_1(x)
    output = self.conv_blk_2(output)
    output = self.fc(output)

    return output
