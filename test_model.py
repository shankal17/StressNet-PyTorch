"""

Created on: 2/3/2021 00:23 AM

@File: test_scsnet.py
@Author: Xufeng Huang (xufenghuang1228@gmail.com & xfhuang@umich.edu)
@Copy Right: Licensed under the MIT License. 

"""
import torch
from model import SCSNet

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
