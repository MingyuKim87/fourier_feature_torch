<div align="center">

# Fourier Features Let Networks Learn  High Frequency Functions in Low Dimensional Domains
[Mingyu Kim\*](https://mingyukim87.github.io)

This is PyTorch implementation on `1D NTK`, `2D Image regression` and `3D NeRF` experiments described in the [paper](https://arxiv.org/abs/2006.10739).  

<div align="left">

## Abstract

This paper illustrates that passing input points through a simple Fourier feature mapping enables a multilayer perceptron (MLP) to learn high-frequency functions in low-dimensional problem domains. These results shed light on recent advances in computer vision and graphics that achieve state-of-the-art results by using MLPs to represent complex 3D objects and scenes. Using tools from the neural tangent kernel (NTK) literature, we show that a standard MLP fails to learn high frequencies both in theory and in practice. To overcome this spectral bias, we use a Fourier feature mapping to transform the effective NTK into a stationary kernel with a tunable bandwidth. We suggest an approach for selecting problem-specific Fourier features that greatly improves the performance of MLPs for low-dimensional regression tasks relevant to the computer vision and graphics communities.

## Code
We provide several python `py`s for experiments in this [paper](https://arxiv.org/abs/2006.10739). If you want to approach the original codes, please visit [original repository](https://github.com/tancik/fourier-feature-networks/tree/master/Experiments).

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - The directory structure should be orgainzed as follows :

```
Fourier_feature_torch
├── 1D_ntk_opt
│   ├── 1d_ntk_opt_torch_v2.py
│   ├── ab_opt_dict.pt
│   └── utils_torch_ntk_v2.py
├── 2D_image_regression
│   ├── image_regression.py
│   └── Results
├── 3D_simple_nerf
│   ├── 3D_simple_nerf.py
│   ├── download_lego.sh
├── FFT_practice
│   ├── example_fft.py
│   └── Results
└── README.md
```


## 1D NTK experiments
For the 1D NTK experiment, this code can be executed as follows:

```Python
python 1d_ntk_opt_torch_v2.py
```

After finished this code, this code outputs two figures. `supp_opt_torch.png` provides function values of NTK kernels varying parameters of fourier feature. 

```
│   ├── Results
│   │   ├── supp_opt_torch.png
│   │   └── torch_temp_opt_fam_p2.0.png
```

## 1D NTK experiments
For the 1D NTK experiment, this code can be executed as follows:

```python
cd 1D_ntk_opt
python 1d_ntk_opt_torch_v2.py
```

After finished this code, this code outputs two figures. `supp_opt_torch.png` provides function values of NTK kernels varying parameters of fourier feature. 

```
│   ├── Results
│   │   ├── supp_opt_torch.png
│   │   └── torch_temp_opt_fam_p2.0.png
```


## 2D image regression experiments
For the 2D image regression, this code can be executed as follows:

```python
cd 2D_image_regression
python image_regression.py
```

After finished this code, this code outputs both animated image and original image .

```
│   ├── Results
│       ├── MLP_10000_basic_1
│       │   ├── generated_img_1.mp4
│       │   └── original_img.png
│       ├── MLP_10000_gauss_1
│       │   ├── generated_img_1.mp4
│       │   └── original_img.png
│       ├── MLP_10000_gauss_10
│       │   ├── generated_img_10.mp4
│       │   └── original_img.png
│       ├── MLP_10000_gauss_100
│       │   ├── generated_img_100.mp4
│       │   └── original_img.png
│       └── MLP_10000_no_-1
│           ├── generated_img_-1.mp4
│           └── original_img.png
```

## 3D NeRF experiments
For the 3D NeRF, this code can be executed as follows:

```python
cd 3D_simple_nerf
sh download_lego.sh
python image_regression.py
```

After finished this code, this code outputs bat_plot, validation image and created image by NeRF models. In this experiment, we choose one fourier feature among `no_encoding`, `basic`, `position_enc`, `position_enc_new` and `gaussian features`

```
│   └── Result
│       ├── bar_plot.png
│       ├── test_image.png
│       └── test.png
```

## Acknowledgements

- This code is migrated from the `jax` code by Matthew Tancik et al., [this repository](https://github.com/tancik/fourier-feature-networks/tree/master/Experiments).

