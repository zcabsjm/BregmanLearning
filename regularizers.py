import torch
import math

class reg_none:
    def __call__(self, x):
        return 0
    
    def prox(self, x, delta=1.0):
        return x
    
    def sub_grad(self, v):
        return torch.zeros_like(v)
    
class reg_l1:
    def __init__(self, lamda=1.0):
        self.lamda = lamda
        
    def __call__(self, x):
        return self.lamda * torch.norm(x, p=1).item()
        
    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.lamda),min=0)
        
    def sub_grad(self, v):
        return self.lamda * torch.sign(v)
    

class reg_l1_pos:
    def __init__(self, lamda=1.0):
        self.lamda = lamda
        
    def __call__(self, x):
        return self.lamda * torch.norm(x, p=1).item()
        
    def prox(self, x, delta=1.0):
        return torch.clamp(torch.sign(x) * torch.clamp(torch.abs(x) - (delta * self.lamda),min=0),min=0)
        
    def sub_grad(self, v):
        return self.lamda * torch.sign(v)
    
    
    
class reg_l1_l2:
    def __init__(self, lamda=1.0):
        self.lamda = lamda
        
    #ToDo: incorporate lamda in call
    def __call__(self, x):
        return self.lamda * math.sqrt(x.shape[-1]) * torch.norm(torch.norm(x,p=2,dim=1), p=1).item()
        
    def prox(self, x, delta=1.0):
        thresh = delta*self.lamda
        thresh *= math.sqrt(x.shape[-1])
        
        ret = torch.clone(x)
        nx = torch.norm(x,p=2,dim=1).view(x.shape[0],1)       
        
        ind = torch.where((nx!=0))[0]
        
        ret[ind] = x[ind] * torch.clamp(1 - torch.clamp(thresh/nx[ind], max=1), min=0)
        return ret
    
        
    def sub_grad(self, x):
        thresh = self.lamda * math.sqrt(x.shape[-1])
        #
        nx = torch.norm(x,p=2,dim=1).view(x.shape[0],1)      
        ind = torch.where((nx!=0))[0]
        ret = torch.clone(x)
        ret[ind] = x[ind]/nx[ind]
        return thresh * ret
    
# subclass for convolutional kernels
class reg_l1_l2_conv(reg_l1_l2):
    def __init__(self, lamda=1.0):
        super().__init__(lamda = lamda)
        
    def __call__(self, x):
        return super().__call__(x.view(x.shape[0]*x.shape[1],-1))
    
    def prox(self, x, delta=1.0):
        ret = super().prox(x.view(x.shape[0]*x.shape[1],-1), delta)
        return ret.view(x.shape)
    
    def sub_grad(self, x):
        ret = super().sub_grad(x.view(x.shape[0]*x.shape[1],-1))
        return ret.view(x.shape) 
                

        
class reg_l1_l1_l2:        
    def __init__(self, lamda=1.0):
        self.lamda = lamda
        #TODO Add suitable normalization based on layer size
        self.l1 = reg_l1(lamda=self.lamda)
        self.l1_l2 = reg_l1_l2(lamda=self.lamda)
        
    def __call__(self, x):
        return 0
        
    def prox(self, x, delta=1.0):
        thresh = delta * self.lamda
                
        return self.l1_l2.prox(self.l1.prox(x,thresh), thresh)
    
    def sub_grad(self, x):
        return self.lamda * (self.l1.sub_grad(x) + self.l1_l2.sub_grad(x))
    
class reg_soft_bernoulli:
    def __init__(self,lamda=1.0):
        self.lamda = lamda
        
    def prox(self, x, delta=1.0):
        return torch.sign(x) * torch.max(torch.clamp(torch.abs(x) - (delta * self.lamda),min=0),torch.bernoulli(0.01*torch.ones_like(x)))
    
    def sub_grad(self, v):
        return self.lamda * torch.sign(v)
    

class reg_nuclear_conv:
    """
    Applies a nuclear norm penalty to convolution weights.
    This flattens conv weights into a 2D matrix, applies SVD,
    and performs singular-value soft-thresholding.
    """
    def __init__(self, lamda=1.0):
        self.lamda = lamda

    def __call__(self, x):
        # Flatten to shape [m, n] = [out_ch*in_ch, kernel_h*kernel_w]
        mat = x.view(x.shape[0]*x.shape[1], -1)
        # Use economy SVD so shapes match
        U, S, V = torch.svd(mat, some=True)
        return self.lamda * torch.sum(S)

    def prox(self, x, delta=1.0):
        # Flatten
        mat = x.view(x.shape[0]*x.shape[1], -1)
        U, S, V = torch.svd(mat, some=True)
        Vh = V.t()

        # Soft-threshold the singular values
        S_thresh = torch.clamp(S - self.lamda * delta, min=0.0)
        
        # Multiply each column of U by the corresponding singular value
        X_thresh = (U * S_thresh.unsqueeze(0)) @ Vh
        
        # Reshape back
        return X_thresh.view(*x.shape)

    def sub_grad(self, x):
        # Subgradient is U @ Vᵀ
        mat = x.view(x.shape[0]*x.shape[1], -1)
        U, S, V = torch.svd(mat, some=True)
        Vh = V.t()
        
        grad = U @ Vh
        return self.lamda * grad.view(*x.shape)

class reg_nuclear_linear:
    def __init__(self, lamda=1.0):
        self.lamda = lamda

    def __call__(self, x):
        # x is [out_features, in_features]
        U, S, V = torch.svd(x, some=True)
        return self.lamda * torch.sum(S)

    def prox(self, x, delta=1.0):
        U, S, V = torch.svd(x, some=True)
        S_thresh = torch.clamp(S - self.lamda * delta, min=0.0)
        return (U * S_thresh.unsqueeze(0)) @ V.t()

    def sub_grad(self, x):
        U, S, V = torch.svd(x, some=True)
        return self.lamda * (U @ V.t())

class reg_nuclear_linear_truncated:
    def __init__(self, lamda=1.0, rank=None, niter=2):
        self.lamda = lamda
        self.rank = rank  # Number of singular values/vectors to compute
        self.niter = niter  # Power iterations for accuracy

    def _svd(self, x):
        if self.rank is not None and self.rank < min(x.shape):
            # Use randomized truncated SVD
            U, S, V = torch.svd_lowrank(x, q=self.rank, niter=self.niter)
        else:
            # Fallback to full SVD
            U, S, V = torch.svd(x, some=True)
        return U, S, V

    def __call__(self, x):
        U, S, V = self._svd(x)
        return self.lamda * torch.sum(S)

    def prox(self, x, delta=1.0):

        # Get SVD
        U, S, V = self._svd(x)
        
        # Calculate threshold
        threshold = self.lamda * delta
        S_thresh = torch.clamp(S - threshold, min=0.0)
        
        # Reconstruction
        return (U * S_thresh.unsqueeze(0)) @ V.t()

    def sub_grad(self, x):
        U, S, V = self._svd(x)
        return self.lamda * (U @ V.t())

class reg_nuclear_conv_truncated:
    def __init__(self, lamda=1.0, rank=None, niter=2):
        self.lamda = lamda
        self.rank = rank
        self.niter = niter

    def _svd(self, x):
        mat = x.view(x.shape[0]*x.shape[1], -1)
        if self.rank is not None and self.rank < min(mat.shape):
            U, S, V = torch.svd_lowrank(mat, q=self.rank, niter=self.niter)
        else:
            U, S, V = torch.svd(mat, some=True)
        return U, S, V, x.shape

    def __call__(self, x):
        U, S, V, shape = self._svd(x)
        return self.lamda * torch.sum(S)

    def prox(self, x, delta=1.0):
        U, S, V, shape = self._svd(x)
        S_thresh = torch.clamp(S - self.lamda * delta, min=0.0)
        X_thresh = (U * S_thresh.unsqueeze(0)) @ V.t()
        return X_thresh.view(*shape)

    def sub_grad(self, x):
        U, S, V, shape = self._svd(x)
        grad = U @ V.t()
        return self.lamda * grad.view(*shape)