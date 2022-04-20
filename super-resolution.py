#!/usr/bin/env python
# coding: utf-8

# Code for **super-resolution** (figures $1$ and $5$ from main paper).. Change `factor` to $8$ to reproduce images from fig. $9$ from supmat.
# 
# You can play with parameters and see how they affect the result. 

# In[ ]:


"""
*Uncomment if running on colab* 
Set Runtime -> Change runtime type -> Under Hardware Accelerator select GPU in Google Colab 
"""
# !git clone https://github.com/DmitryUlyanov/deep-image-prior
# !mv deep-image-prior/* ./


# # Import libs

# In[ ]:


from __future__ import print_function
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')

import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from models import *

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models.downsampler import Downsampler

from utils.sr_utils import *
from pdb import *
from torch.nn import functional as F
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize = -1 
factor = 4 # 8
enforse_div32 = 'CROP' # we usually need the dimensions to be divisible by a power of two (32 in this case)
PLOT = True

# To produce images from the paper we took *_GT.png images from LapSRN viewer for corresponding factor,
# e.g. x4/zebra_GT.png for factor=4, and x8/zebra_GT.png for factor=8 
path_to_image = 'data/sr/zebra_GT.png'


# # Load image and baselines

# In[ ]:


# Starts here
imgs = load_LR_HR_imgs_sr(path_to_image , imsize, factor, enforse_div32)

imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np'] = get_baselines(imgs['LR_pil'], imgs['HR_pil'])
# PLOT = False
if PLOT:
    # plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], imgs['sharp_np'], imgs['nearest_np']], 4,12);
    print ('PSNR bicubic: %.4f   PSNR nearest: %.4f' %  (
                                        compare_psnr(imgs['HR_np'], imgs['bicubic_np']), 
                                        compare_psnr(imgs['HR_np'], imgs['nearest_np'])))


# # Set up parameters and net

# In[ ]:


input_depth = 32
 
INPUT =     'noise'
pad   =     'reflection'
OPT_OVER =  'net'
KERNEL_TYPE='lanczos2'

LR = 0.1
tv_weight = 0.0

OPTIMIZER = 'adam'
OPTIMIZER = 'LBFGS'

if factor == 4: 
    num_iter = 2000
    reg_noise_std = 0.03
elif factor == 8:
    num_iter = 4000
    reg_noise_std = 0.05
else:
    assert False, 'We did not experiment with other factors'


# In[ ]:


net_input_noise = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()
# set_trace()
NET_TYPE = 'skip' # UNet, ResNet
n_channels=32


class ConvSplitTree2(nn.Module):
    def __init__(self, tree_depth, in_channels, out_channels= 2, kernel_size=3, stride=1, pad=1, dilation=1, guide_in_channels =1, n_split=2):
        super(ConvSplitTree2, self).__init__()
        if tree_depth>6:
            tree_depth = 6
        self.tree_depth = tree_depth 
        self.n_split = n_split
        self.tree_nodes = (tree_depth * n_split)
        self.sum_out_channels = int(self.tree_nodes * out_channels)
        self.convSplit = nn.Conv2d(guide_in_channels, self.sum_out_channels, kernel_size=1, stride=stride, padding=0, bias=False)#.type(dtype)
        nn.init.uniform_(self.convSplit.weight)
        self.out_channels = out_channels
        self.convPred = nn.Conv2d(in_channels, self.sum_out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=True)#.type(dtype)
        nn.init.uniform_(self.convPred.weight)
        self.normalize_weight_iter = False
        self.kernel_size = kernel_size
        self.resize_small = 1

    def forward(self,x, data):
        if x.shape[2]< data.shape[2]:
            if self.resize_small ==1:
                data = F.interpolate(data, x.size()[2:], mode='bilinear', align_corners=True)
            else:
                x = F.interpolate(x, data.size()[2:], mode='bilinear', align_corners=True)

        if x.shape[2]> data.shape[2]:
            if self.resize_small ==1:
                x = F.interpolate(x, data.size()[2:], mode='bilinear', align_corners=True)
            else:
                data = F.interpolate(data, x.size()[2:], mode='bilinear', align_corners=True)
        # set_trace()
        score =self.convSplit(x).view(x.shape[0],self.tree_depth,self.n_split, self.out_channels,x.shape[2],x.shape[3])
        data = self.convPred(data).view(x.shape[0],self.tree_depth,self.n_split,self.out_channels,x.shape[2],x.shape[3])
        score = F.softmax(score, dim=2)
        data = torch.sum(torch.sum(score * data, dim=2),dim=1)
        final_score,_ = torch.max(score, dim=2)
        final_score = torch.prod(final_score, dim=1)
        y= final_score * data
        return y

net = get_net(input_depth, 'skip', pad,
              skip_n33d=n_channels, 
              skip_n33u=n_channels, 
              skip_n11=4, 
              num_scales=3,
              upsample_mode='bilinear').type(dtype)

# Losses
mse = torch.nn.MSELoss().type(dtype)

img_LR_var = np_to_torch(imgs['LR_np']).type(dtype)
# net_input = F.interpolate(img_LR_var, [imgs['HR_pil'].size[1], imgs['HR_pil'].size[0]], mode='bilinear', align_corners=True).detach().clone()

net_input = get_noise(input_depth, INPUT, (imgs['HR_pil'].size[1], imgs['HR_pil'].size[0])).type(dtype).detach()

downsampler = Downsampler(n_planes=3, factor=factor, kernel_type=KERNEL_TYPE, phase=0.5, preserve_size=True).type(dtype)
cst = ConvSplitTree2(tree_depth=2, in_channels=input_depth, out_channels= 3,  kernel_size=3, stride=1, pad=1, dilation=1, guide_in_channels = input_depth, n_split=2)
cst = cst.type(dtype)
# # Define closure and optimize

# In[ ]:


def closure():
    global i, net_input
    
    # if reg_noise_std > 0:
    #     net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    # out_HR = net(net_input)
    out_HR = cst(net_input_noise, net_input)
    # out_HR += net_input
    out_LR = downsampler(out_HR)

    total_loss = mse(out_LR, img_LR_var) 
    
    if tv_weight > 0:
        total_loss += tv_weight * tv_loss(out_HR)
        
    total_loss.backward()

    # Log
    psnr_LR = compare_psnr(imgs['LR_np'], torch_to_np(out_LR))
    psnr_HR = compare_psnr(imgs['HR_np'], torch_to_np(out_HR))
    if i % 10 == 0: print ('Iteration %05d    PSNR_LR %.3f   PSNR_HR %.3f' % (i, psnr_LR, psnr_HR), '\r', end='')
                      
    # History
    psnr_history.append([psnr_LR, psnr_HR])
    
    if PLOT and i % 100 == 0:
        out_HR_np = torch_to_np(out_HR)
        plot_image_grid([imgs['HR_np'], imgs['bicubic_np'], np.clip(out_HR_np, 0, 1)], factor=13, nrow=3)

    i += 1
    
    return total_loss


# In[ ]:


psnr_history = [] 
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()

i = 0
OPT_OVER =  'net,input'
p = get_params(OPT_OVER, cst, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


# In[ ]:


out_HR_np = np.clip(torch_to_np(net(net_input)), 0, 1)
result_deep_prior = put_in_center(out_HR_np, imgs['orig_np'].shape[1:])

# For the paper we acually took `_bicubic.png` files from LapSRN viewer and used `result_deep_prior` as our result
plot_image_grid([imgs['HR_np'],
                 imgs['bicubic_np'],
                 out_HR_np], factor=4, nrow=1);


# In[ ]:




