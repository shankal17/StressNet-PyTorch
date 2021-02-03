"""

Created on: 2/3/2021 00:18 AM

@File: model.py
@Author: Xufeng Huang (xufenghuang1228@gmail.com & xfhuang@umich.edu)
@Copy Right: Licensed under the MIT License. 
@Ref: [1] Z. Nie, H. Jiang, L.B. Kara, Stress Field Prediction in Cantilevered Structures Using Convolutional Neural Networks,
      Journal of Computing and Information Science in Engineering. 20 (2019). https://doi.org/10.1115/1.4044097.

"""
# The first architecture is a single-channel stress prediction
# neural network (SCSNet) where the loads are augmented
# with the feature representation (FR). The second is a multichannel
# stress prediction neural network (StressNet)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSNet(nn.Module):
    """The single-channel stress prediction neural network (SCSNet)

    """

    def __init__(self):
        super(SCSNet, self).__init__()
        down_kwargs = dict(stride=1, padding=1)
        up_kwargs = dict(stride=2, padding=1, output_padding=1)

        self.conv1 = nn.Conv2d(1, 32, 3, **down_kwargs)
        self.conv2 = nn.Conv2d(32, 64, 3, **down_kwargs)
        self.convUp1 = nn.ConvTranspose2d(64, 64, 3, **up_kwargs)
        self.conv3 = nn.Conv2d(64, 32, 3, **down_kwargs)
        self.convUp2 = nn.ConvTranspose2d(32, 32, 3, **up_kwargs)
        self.conv4 = nn.Conv2d(32, 16, 3, **down_kwargs)
        self.conv5 = nn.Conv2d(16, 1, 3, **down_kwargs)
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 30)
        self.fc3 = nn.Linear(32, 1024)
        self.fc4 = nn.Linear(1024, 3072)

        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, input1, input2):
        x = self.pool(F.relu(self.conv1(input1)))  # E1+E2
        x = self.pool(F.relu(self.conv2(x)))  # E3+E4
        x = x.view(-1, 64 * 6 * 8)  # E5
        x = self.dropout(F.relu(self.fc1(x)))  # E6
        x = F.softplus(self.fc2(x))  # E7
        x = torch.cat([x, input2], 1)  # FR

        x = F.softplus(self.fc3(x))  # D1
        x = self.dropout(F.relu(self.fc4(x)))  # D2
        x = x.view(-1, 64, 6, 8)  # D3
        x = self.convUp1(x)  # D4
        x = F.relu(self.conv3(x))  # D5
        x = self.convUp2(x)  # D6
        x = F.relu(self.conv4(x))  # D7
        x = F.relu(self.conv5(x))  # D7
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size, height, width = 256, 24, 32
    sim_node = torch.randn(batch_size, 1, height, width).to(device)
    sim_load = torch.randn(batch_size, 2).to(device)

    # test SCSNet
    scsnet_model = SCSNet()
    scsnet_model.cuda()
    output_feature = scsnet_model(sim_node, sim_load)
    print(output_feature.size())