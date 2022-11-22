import os
import sys
import random
import numpy as np
import torch

from torch import optim, nn

import time
import json
import copy

from livelossplot import PlotLosses
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
import matplotlib
import matplotlib.pylab as pylab
from matplotlib.lines import Line2D

# device
device_num = int(float('1'))
torch.cuda.set_device(device_num)
device = torch.device('cuda') \
    if torch.cuda.is_available() else torch.device('cpu')

# ntk
from utils_torch_ntk_v2 import *

# function
def fplot(x):
  output = torch.fft.fftshift(torch.log10(torch.abs(torch.fft.fft(x)))).detach().cpu().numpy()
  return output


# function : position encoding 
input_encoder = lambda x, a, b: \
    torch.cat([a * torch.sin((2.*torch.pi*x[...,None]) * b), \
        a * torch.cos((2.*torch.pi*x[...,None]) * b)], axis=-1) / torch.linalg.norm(a) 

# function : tile
alias_tile = lambda f, N, M : \
    torch.tile(f.view(list(f.shape[:-1])+[M,N]).mean((-2)), [1]*len(f.shape[:-1])+[M])

# Data pipeline
  # Signal makers
def sample_random_signal(decay_vec):
  # device
  device = decay_vec.device
  
  N = decay_vec.shape[0]
  raw = torch.randn(N, 2, device=device) @ torch.ones(2, device=device)
  signal_f = raw * decay_vec
  signal = torch.real(torch.fft.ifft(signal_f))
  return signal

def sample_random_powerlaw(N, power, device):
  coords = torch.fft.ifftshift(1 + N//2 - \
      torch.abs(torch.fft.fftshift(torch.arange(N)) - N//2)).float().to(device)
  decay_vec = coords ** -power
  return sample_random_signal(decay_vec)

def make_network(num_layers, num_input, num_channels, num_outputs=1):
    # container
    layers = torch.nn.ModuleList()
    
    for i in range(num_layers-1):
        if i == 0:
            # MLP   
            layers.append(nn.Linear(num_input, num_channels))
        else:
            # MLP   
            layers.append(nn.Linear(num_channels, num_channels))
        
        # Activation
        layers.append(nn.ReLU())
        
    # Head layers
    layers.append(nn.Linear(num_channels, num_outputs))
    
    # model
    model = nn.Sequential(*layers)
    
    return model
  
def func_model(model):
    fnet, params = make_functional(model)
    
    ntk_kernel_fn = lambda x1, x2 : \
      empirical_ntk_jacobian_contraction(fnet, params, x1, x2, compute='trace')
      
    return fnet, params, ntk_kernel_fn
  
def compute_ntk(x, avals, bvals, kernel_fn):
    # device
    device = x.device
    
    # position encoding
    x1_enc = input_encoder(x, avals, bvals)
    x2_enc = input_encoder(torch.tensor([0.]).float().to(device), avals, bvals)
    
    # For DEBUG
    tk = kernel_fn(x1_enc, x2_enc)
    
    # For DEBUG
    out = torch.squeeze(tk)
    
    return out

### Prediction with fft and inverse fft
def predict(kernel_fn, yf_test, pred0f_test, ab, t_final, device, eta=None):
  # dimension
  N, M = yf_test.shape[-1]//2, 2

  # define a domain
  x_test = torch.linspace(0., 1., N*M).to(device)
  
  # construction ntk 
    # using features, parameters with position encoding, definition of kernel_fn
  H_row_test = compute_ntk(x_test,  *ab, kernel_fn)
  
  # transmit H_row_test (using kernel matrix) do frequency domain
  H_t = torch.real(torch.fft.fft(H_row_test))
  
  # tiling
  H_d_tile = alias_tile(H_t, N, M)
  
  # numerical stability
  if eta is None:
    H_d_tile_train = 1. - torch.exp(-t_final * H_d_tile)
  else:
    H_d_tile_train = 1. - (1. - eta * H_d_tile) ** t_final
    
  # ???
  yf_train_tile = alias_tile(yf_test, N, M)  
  pred0f_train_tile = alias_tile(pred0f_test, N, M)
  
  # ???
  exp_term = H_d_tile_train * (yf_train_tile - pred0f_train_tile)
  pred_train = (pred0f_train_tile + exp_term)[...,:N]
  pred_test = pred0f_test + H_t / H_d_tile * exp_term

  pred_test = torch.real(torch.fft.ifft(pred_test))[1::2]
  pred_train = torch.real(torch.fft.ifft(pred_train))

  return pred_test, pred_train

def predict_psnr(kernel_fn, y_test, pred0f_test, ab, t_final, device, eta=None):
  yf_test = torch.fft.fft(y_test)
  N, M = yf_test.shape[-1]//2, 2
  pred_test, pred_train = predict(kernel_fn, yf_test, pred0f_test, ab, t_final, device, eta)
  calc_psnr = lambda f, g : -10.*torch.log10(torch.mean(torch.abs(f - g)**2, -1)).mean()
  return calc_psnr(y_test[1::2], pred_test), calc_psnr(y_test[::2], pred_train)

def optimize(network_size, y_test, y_gt, t_final, ab_init, name, \
    kernel_lr=.01, iters=800, device=torch.device('cpu')):
    
  network = make_network(*network_size).to(device)
  fnet, params, ntk_fn = func_model(network)
  
  def spectrum_loss(params):
    pred_test = predict(ntk_fn, torch.fft.fft(y_test), torch.zeros_like(y_test),\
        (params[0], ab_init[1]), t_final, device=device)[0]
    
    numerator = 0.5 * (torch.abs(y_test[1::2] - pred_test)**2).mean()
    # denominator = torch.prod(y_test.shape[-1:].float()) 
    denominator = torch.numel(y_test[0])
    
    output = numerator / denominator
    return output 
    # return .5 * np.mean(np.abs(y_test[1::2] - pred_test)**2) / np.prod(y_test.shape[-1:])

  # optimizer for torch
  a_noise = torch.abs(torch.randn(*ab_init[0].shape, device=device)) * .001
  opt = optim.Adam((ab_init[0] + a_noise, ab_init[1]), lr=kernel_lr)
  
  groups = {'losses {}'.format(name): ['curr_test', 'gt_test', 'gt_train'],}
  plotlosses = PlotLosses(groups=groups)
  print('begin')
  
  # initialization for alpha and beta
    # params
  init_ab = copy.deepcopy(opt.param_groups[-1]['params'])
  ab = [param.requires_grad_(True) for param in opt.param_groups[-1]['params']]
  print("ab : ", init_ab)

  for i in range(iters):
      # initialization grad
      opt.zero_grad()
          
      # eval loss
      loss = spectrum_loss(ab) if i != 0 else spectrum_loss(init_ab)
      
      # compute grad
      loss.requires_grad_(True)
      loss.backward()
      
      # update
      opt.step()
      
      # new params
      ab = opt.param_groups[-1]['params']
      
      gt_vals = predict_psnr(ntk_fn, y_gt, torch.zeros_like(y_gt),\
          ab, t_final, device=device)

      plotlosses.update({'curr_test':-10.*torch.log10(2.*loss).item(), 
                         'gt_test': gt_vals[0].item(),
                         'gt_train': gt_vals[1].item(),}, current_step=i)
      if i % 20 == 0:
          plotlosses.send()

  optimized_params = opt.param_groups[-1]['params']
  avals_optimized = optimized_params[0]
  return avals_optimized


# Hyper-parameter
N_train = 128
data_powers = [1.0, 1.5, 2.0]
N_test_signals = 8
N_train_signals = 1024

network_size = (4, 1024)
network_size_torch = (4, N_train, 1024)

learning_rate = 5e-3
sgd_iters = 1000

if __name__ == "__main__":    
    '''
        Phase 1 : Testing optimal \alpha and \beta values
          for plotting, {train, test} datasets are identical
    '''
    np.random.seed(0)
    torch.manual_seed(0)

    # Signal
    M = 2
    N = N_train
    x_test = torch.linspace(0,1.,N*M).float().to(device)
    x_train = x_test[::M]


    def data_maker(N_pts, N_signals, p, device):
        # data_np = np.stack([sample_random_powerlaw(N_pts, p) for i in range(N_signals)])
        data = torch.stack([sample_random_powerlaw(N_pts, p, device) for i in range(N_signals)])
        data = (data - data.min()) / (data.max() - data.min())  - .5
        return data


    s_dict = {}
    for p in data_powers:
        ret = data_maker(N*M, N_test_signals, p, device)
        s_dict['data_p{}'.format(p)] = ret


    # Kernels
    bvals = torch.arange(1, N//2+1).float().to(device)
    ab_dict = {}
    powers = torch.linspace(0, 2, 17).float().to(device)
    ab_dict.update({'power_{}'.format(p) : (bvals**-p, bvals) for p in powers})
    ab_dict['power_infty'] = (torch.eye(bvals.shape[0], device=device)[0], bvals)


    # optimize
    ab_opt_dict = {}
    t_final = learning_rate * sgd_iters

    
    for p in data_powers:
        k = 'opt_fam_p{}'.format(p)
        ab = ab_dict['power_infty']
        train_signals = data_maker(N*M, N_train_signals, p, device)
        a_opt = optimize(network_size_torch, train_signals, \
            s_dict['data_p{}'.format(p)][0], t_final, ab, k, iters=800, device=device)

        ab_opt_dict[k] = (a_opt, bvals)
    
    
    ab_dict.update(ab_opt_dict)
    
    # save (file i/o)
    torch.save(ab_dict, "./1d_ntk_opt/ab_opt_dict.pt")

    for k in ab_opt_dict:
      plt.plot(np.abs(ab_opt_dict[k][0].detach().cpu().numpy()), label=k)
    plt.legend()
    plt.savefig("./torch_temp_{}.png".format(k))
    plt.clf()
    
    
    """
      Phase 2 : 
    """
    
    # file (i/o)
    ab_dict = torch.load("ab_opt_dict.pt", map_location=device)
    
    # kernel_fn <- for "NTK" come from make_network(*(network_size))[-1]
    H_rows = {k : compute_ntk(x_train, *ab_dict[k], func_model(make_network(*network_size_torch).to(device))[-1]) \
      for k in ab_dict}
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    params = {'legend.fontsize': 22,
            'axes.labelsize': 24,
            'axes.titlesize': 28,
            'xtick.labelsize':22,
            'ytick.labelsize':22}
    pylab.rcParams.update(params)


    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    matplotlib.rcParams['mathtext.rm'] = 'serif'

    plt.rcParams["font.family"] = "cmr10"


    N_fams = len(data_powers)


    fig3 = plt.figure(constrained_layout=True, figsize=(20,6))
    gs = fig3.add_gridspec(1,2)

    colors_k = np.array([[0.8872, 0.4281, 0.1875],
        # [0.8136, 0.6844, 0.0696],
        [0.2634, 0.6634, 0.4134],
        [0.0943, 0.5937, 0.8793],
        [0.3936, 0.2946, 0.6330],
        [0.7123, 0.2705, 0.3795]])

    linewidth = 4
    legend_offset = -.27

    fams = ['1.0', '1.5', '2.0']
    # power_infinity = compute_ntk(x_train, *ab_dict['power_infty'], func_model(make_network(*network_size_torch).to(device))[-1])
    # plt.semilogy(10**fplot(power_infinity), color='k', label=fr'Power $\infty$', linewidth=linewidth, linestyle='-', alpha=.6)
    idx = 0
    for i, k in enumerate(H_rows):
      if 'opt' in k:
        plt.semilogy(10**fplot(H_rows[k]), color=colors_k[idx], label=fr'Opt. for $\alpha={fams[idx]}$', linewidth=linewidth, alpha=.8)
        idx += 1
      else:
        plt.semilogy(10**fplot(H_rows[k]), label='Power_{}'.format(k), linewidth=linewidth, alpha=.8)
    plt.title('(a) NTK Fourier spectrum', y=legend_offset)
    plt.xticks([0,32,64,96,128], ['$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'])

    plt.legend(loc='upper right', prop={'size': 6})
    plt.xlabel('Frequency')
    plt.ylabel(r'Magnitude')
    plt.grid(True, which='major', alpha=.3)
    plt.savefig('supp_opt_torch.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
