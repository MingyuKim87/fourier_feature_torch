import torch
import torch.nn as nn
from functorch import make_functional, vmap, vjp, jvp, jacrev

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full'):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = [j.flatten(2) for j in jac1] # for flattening networks -> 

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = [j.flatten(2) for j in jac2]
    
    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False
        
    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

def empirical_ntk_ntk_vps(func, params, x1, x2, compute='full'):
    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes vec @ J(x2).T
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes J(X1) @ vjps
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)
        
    # get_ntk(x1, x2) computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to empirical_ntk_ntk_vps are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the vmaps here do.
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)
    
    if compute == 'full':
        return result
    if compute == 'trace':
        return torch.einsum('NMKK->NM', result)
    if compute == 'diagonal':
        return torch.einsum('NMKK->NMK', result)

if __name__ == "__main__":
    device = torch.device("cuda")
    
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, (3, 3))
            self.conv2 = nn.Conv2d(32, 32, (3, 3))
            self.conv3 = nn.Conv2d(32, 32, (3, 3))
            self.fc = nn.Linear(21632, 10)
            
        def forward(self, x):
            x = self.conv1(x)
            x = x.relu()
            x = self.conv2(x)
            x = x.relu()
            x = self.conv3(x)
            x = x.flatten(1)
            x = self.fc(x)
            return x
        
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(1, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 1024)
            self.fc4 = nn.Linear(1024, 1)
            
        def forward(self, x):
            x = self.fc1(x)
            x = x.relu()
            x = self.fc2(x)
            x = x.relu()
            x = self.fc3(x)
            x = x.relu()
            x = self.fc4(x)
            return x
        
    x_cnn_train = torch.randn(20, 3, 32, 32, device=device)
    x_cnn_test = torch.randn(5, 3, 32, 32, device=device)
    
    x_mlp_train = torch.randn(20, 1, device=device)
    x_mlp_test = torch.randn(20, 1, device=device)
    
    cnn_net = CNN().to(device)
    mlp_net = MLP().to(device)
    fnet, params = make_functional(mlp_net)
    
    def fnet_single(params, x):
        return fnet(params, x)
    
    # option 1 : jacobian contraction
        # compute : full
    result = empirical_ntk_jacobian_contraction(fnet_single, params, x_mlp_train, x_mlp_test)
    print(result.shape)
    
        # compute : trace
    result = empirical_ntk_jacobian_contraction(fnet_single, params, x_mlp_train, x_mlp_test, 'trace')
    print(result.shape)
    
    # option 2 : NTK-vector product
    result_from_jacobian_contraction = empirical_ntk_jacobian_contraction(fnet_single, params, x_mlp_test, x_mlp_train)
    result_from_ntk_vps = empirical_ntk_ntk_vps(fnet_single, params, x_mlp_test, x_mlp_train)
    
    print(result_from_jacobian_contraction.shape)
    print(result_from_ntk_vps.shape)
    
    print(torch.allclose(result_from_jacobian_contraction, result_from_ntk_vps, atol=1e-5))
    
    
    