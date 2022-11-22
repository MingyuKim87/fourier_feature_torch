import time
import os
import sys
from unittest import expectedFailure
import json

import torch
import torch.nn as nn
from tqdm import tqdm

import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

class GaussianFourierFeatureTransform(nn.Module):
    def __init__(self, type, input_dim, mapping_dim, scale, device):
        super().__init__()
        
        self.device = device
        self.type = type
        self.input_dim = input_dim
        self.mapping_dim = self.get_mapping_dim(mapping_dim)
        self.scale = scale
        self.preprocessing_param = self.get_preprocessing_param() * self.scale 
        


    def get_preprocessing_param(self):
        assert self.type in ["no", "basic", "gauss"], "must define pre-processing"
        
        if self.type.find("gauss") >= 0:
            param = torch.randn(self.input_dim, self.mapping_dim // 2).to(self.device)
        elif self.type == "basic":
            param = torch.eye(self.input_dim).to(self.device)
        else:
            param = -1
        
        return param
    
    def forward(self, x):
        x_transform = 2 * torch.pi * torch.matmul(x, self.preprocessing_param) if self.type != "no" else None
        return torch.cat([torch.sin(x_transform), torch.cos(x_transform)], dim=-1) if self.type != "no" else x
    
    def get_mapping_dim(self, mapping_dim):
        if self.type == "no":
            mapping_dim = self.input_dim
        elif self.type == "basic":
            mapping_dim = self.input_dim * 2
        elif self.type.find("gauss") >= 0:
            mapping_dim = mapping_dim
            
        return mapping_dim
        
def tensor_to_numpy(tensor):
    tensor = tensor * 256
    tensor[tensor > 255] = 255
    tensor[tensor < 0 ] = 0
    tensor = tensor.type(torch.uint8).cpu().numpy()
    
    return tensor

def get_image():
    image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
    img = imageio.v2.imread(image_url)[..., :3] / 255.
    c = [img.shape[0] // 2, img.shape[1] // 2]
    r = 256
    img = img[c[0] - r:c[0] + r, c[1] - r:c[1] + r]
    
    return img

def get_conv_models(device):
    model = nn.Sequential(
        nn.Conv2d(2,256, kernel_size=1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),

        nn.Conv2d(256,256,kernel_size=1, padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),

        nn.Conv2d(256,256,kernel_size=1,padding=0),
        nn.ReLU(),
        nn.BatchNorm2d(256),

        nn.Conv2d(256,3,kernel_size=1,padding=0),
        nn.Sigmoid(),

    ).to(device)
    
    return model
    
def get_mlp_models(device, input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim,256),
        nn.ReLU(),

        nn.Linear(256,256),
        nn.ReLU(),

        nn.Linear(256,256),
        nn.ReLU(),

        nn.Linear(256,3),
        nn.Sigmoid(),

    ).to(device)
    
    return model

def get_outputs(step, loss, predicts, targets):
    with torch.no_grad():
        l1_loss = torch.nn.functional.l1_loss(targets, predicts)
        l2_loss = torch.nn.functional.mse_loss(targets, predicts)
        psnr_loss = -10 * torch.log10(2. * l2_loss)
    
    return {
        'steps' : step, "train_loss" : loss.item(), 'l1_loss': l1_loss.item(), 'l2_loss' : l2_loss.item(),\
        'psnr_loss' : psnr_loss.item(), 'output' : tensor_to_numpy(predicts[0])
    }
    
    
if __name__ == "__main__":
    # device
    device = torch.device("cuda")
    
    # preprocessing (fourier feature)
    type = 'gauss'
    scale = 100 if type != 'no' else -1
    preprocessing = GaussianFourierFeatureTransform(type, 2, 256, scale, device)
    
    # iters
    iters = 10000
    
    # experimental name
    experiment_name = "MLP_" + str(iters) + "_" + preprocessing.type + "_" + str(scale)
    
    # result containers
    result_list = []
    
    # property
    is_conv = False
    
    # directory
    pwd = os.getcwd()
    workspace = os.path.join(pwd, experiment_name)
    os.makedirs(workspace, exist_ok=True)
    
    # Get an image that will be the target for our model.
    target = torch.tensor(get_image()).unsqueeze(0).to(device)
    plt.imshow(tensor_to_numpy(target[0]))
    plt.savefig(os.path.join(workspace, "original_img.png"))
    
    # Create input pixel coordinates in the unit square. This will be the input to the model.
    coords = np.linspace(0, 1, target.shape[2], endpoint=False)
    
    # data preprocessing
    if is_conv:
        xy_grid = np.stack(np.meshgrid(coords, coords), -1) # [512, 512] \in { {0, ..., 1}, {0, ..., 1}}
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(device) # contiguous : rearrange a tensor 
        
    else:
        xy_grid = np.stack(np.meshgrid(coords, coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).float().contiguous().to(device)
        xy_grid_train = xy_grid[:, 0:511:2, 0:511:2, :].contiguous()
        
        # flatten
        xy_grid = xy_grid.view(-1, 2)
        xy_grid_train = xy_grid_train.view(-1, 2)
        
    # train / test split
    target_train = target[:, 0:511:2, 0:511:2, :]
    
    # model
    model = get_mlp_models(device, preprocessing.mapping_dim)
    
    # optimizer
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4)
    
    # training
    for epoch in tqdm(range(iters)):
        optimizer.zero_grad()
        
        transformed_x_train = preprocessing(xy_grid_train)
        generated_train = model(transformed_x_train).view(1, 256, 256, 3)
        
        loss = torch.nn.functional.mse_loss(target_train.float(), generated_train)
        # loss = torch.nn.functional.l1_loss(target.float(), generated)
        
        loss.backward()
        optimizer.step()
        
        # test
        with torch.no_grad():
            transformed_x = preprocessing(xy_grid)
            generated = model(transformed_x).view(1, 512, 512, 3)
        
        if epoch % 25 == 0:
            # outputs
            print('Epoch {:d}, loss = {:.3f}'.format(epoch, float(loss)))
            result_list.append(get_outputs(epoch, loss, generated, target))
            
    # animation results
    all_preds = np.concatenate([result['output'][None, :] for result in result_list], axis=0)
    data8 = (np.clip(all_preds, 0, 255)).astype(np.uint8)
    imageio.mimwrite(os.path.join(workspace, 'generated_img_{}.mp4'.format(str(scale))), data8, fps=20)
    
    # save result
    with open(os.path.join(workspace, 'outputfile.json', 'w')) as file:
        json.dump(result_list, file)
            
        
    
    