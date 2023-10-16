#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 12:31:03 2023

@author: isaac
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric
import torch_geometric.nn as gnn

torch.manual_seed(0)  # to see better, you can use a manual seed. 

# if it is "edge_weight"
gcn = gnn.GCNConv(in_channels=18, out_channels=36)

x = torch.randn(22, 18)
edge_index = torch.randint(0, 22, [2, 40])
edge_weight = torch.randn(40)
# NOTE(WMF): I really do not understand the line "above", and I just "try" to get that. 
# in /torch_geometric/utils/loop.py
# if edge_weight is not None:
#   assert edge_weight.numel() == edge_index.size(1)
#   inv_mask = ~mask

y = gcn(x=x, edge_index=edge_index, edge_weight=edge_weight)