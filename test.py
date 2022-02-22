#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : cyl
# @Time : 2022/2/15 17:02 
import torch
import torch.nn.functional as F


input1 = torch.randn(100, 128, 3)
input2 = torch.randn(100, 128, 3)
output = F.pairwise_distance(input1, input2)
print(output)
print(output.shape)