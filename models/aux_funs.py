import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from itertools import cycle
  
    
def init_weight_bias_normal(m):
    if type(m) == nn.Linear:
        m.weight.data = torch.randn_like(m.weight.data)
        m.bias.data = torch.randn_like(m.bias.data)
        
        
        
def sparsify_(model, sparsity, ltype = nn.Linear, conv_group=True, row_group = False):       
    for m in model.modules():
        if not isinstance(m, ltype):
            continue
            
        elif (isinstance(m, nn.Linear) and not row_group) or (isinstance(m, nn.Conv2d) and not conv_group):
            s_loc = sparsity
            mask = torch.bernoulli(s_loc*torch.ones_like(m.weight))
            m.weight.data.mul_(mask)
            
        elif isinstance(m, nn.Linear): # row sparsity
            s_loc = sparsity
            w = m.weight.data
            mask = torch.bernoulli(s_loc*torch.ones(size=(w.shape[0],1),device=w.device))
            #
            m.weight.data.mul_(mask)
            
        elif isinstance(m, nn.Conv2d): # kernel sparsity
            s_loc = sparsity
            w = m.weight.data
            n = w.shape[0]*w.shape[1]
            
            # assign mask
            mask = torch.zeros(n,1,device=w.device)
            idx = torch.randint(low=0,high=n,size=(math.ceil(n*s_loc),))
            mask[idx] = 1
            
            # multiply with mask
            c = w.view(w.shape[0]*w.shape[1],-1)
            m.weight.data = mask.mul(c).view(w.shape)
            
        
# def sparsify_(model, sparsity):
#     if isinstance(sparsity, list):
#         s_iter = cycle(sparsity)
#     else:
#         s_iter = cycle([sparsity])
        
#     for m in model.modules():
#         if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
#             s_loc = next(s_iter)
#             # number of zeros
#             n = int(s_loc*m.weight.numel())
#             # initzialize mask
#             mask = torch.zeros_like(m.weight)
#             row_idx = torch.randint(low=0,high=mask.shape[0],size=(n,))
#             col_idx = torch.randint(low=0,high=mask.shape[1],size=(n,))
#             # fill with ones at random indices
#             mask[row_idx, col_idx] = 1.
#             m.weight.data.mul_(mask)

def sparse_bias_uniform_(model,r0,r1,ltype = nn.Linear):
    for m in model.modules():
        if isinstance(m,ltype):
            if hasattr(m, 'bias') and not (m.bias is None):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound0 = r0 / math.sqrt(fan_in)
                bound1 = r1/math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound0, bound1)
                
def bias_constant_(model,r):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            if type(m) == nn.Linear:
                nn.init.constant_(m.bias, r)           
                
def sparse_weight_normal_(model,r,ltype = nn.Linear):
    for m in model.modules():
        if isinstance(m,ltype):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data.mul_(r)
                

def sparse_weight_uniform_(model,r):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            #nn.init.kaiming_uniform_(m.weight, a=r*math.sqrt(5))
            fan = nn.init._calculate_correct_fan(m.weight, 'fan_in')
            std = r / math.sqrt(fan)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            
            with torch.no_grad():
                m.weight.uniform_(-bound, bound)
    
    

def sparse_he_(model, r):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            if type(m) == nn.Linear:
                w_std = r*math.sqrt(2/(m.weight.data.shape[1]))
                b_std = r*math.sqrt(2/(m.weight.data.shape[1]))

                m.weight.data = torch.normal(0, w_std, size = m.weight.data.shape)
                m.bias.data = b_std *torch.rand(size=m.bias.data.shape)




# def he_sparse_(tensor, sparsity):
#     rows, cols = tensor.shape
#     num_zeros = int(sparsity * rows)

#     with torch.no_grad():
#         tensor.normal_(0, cols*(1-sparsity))
#         for col_idx in range(cols):
#             row_indices = torch.randperm(rows)
#             zero_indices = row_indices[:num_zeros]
#             tensor[zero_indices, col_idx] = 0
#     return tensor


def print_sparsity(M,print_all=True):
    s =""
    s_list =[]
    n=""
    n_list=[]
    sp=0
    numel=0
    for m in M:
        if isinstance(m, torch.nn.Linear):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            
            sp_loc = torch.count_nonzero(a.data).item()
            sp += sp_loc
            s += str(sp_loc/numel_loc) + " "
            s_list.append(sp_loc/numel_loc)
            n += str(torch.count_nonzero(torch.sum(torch.abs(a.data),axis=1)).item()) + "/" + str(a.data.shape[0]) + " "
            n_list.append(torch.count_nonzero(torch.sum(torch.abs(a.data),axis=1)).item()/a.data.shape[0])
        elif isinstance(m, torch.nn.Conv2d):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            
            sp_loc = torch.count_nonzero(a.data).item()
            sp += sp_loc
            s += str(sp_loc/numel_loc) + " "
    
            
    print(50*'-')
    if print_all:
        print('Weight Sparsity:', s)
        print('Active Nodes:', n)
    print('Total percentage of used weights:',(sp/numel))
    
    return s_list, n_list, sp/numel

def net_sparsity(model):
    numel = 0
    nnz = 0
    #
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            nnz += torch.count_nonzero(a.data).item()
    #
    return nnz/numel

def node_sparsity(model):
    ret = []
    #
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            a = m.weight
            
            nnz = torch.count_nonzero(torch.norm(a.data,p=2,dim=1)).item()
            numel_loc = a.shape[0]
            ret.append(nnz/numel_loc)
    #
    return ret

def linear_sparsity(model):
    numel = 0
    nnz = 0
    #
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            a = m.weight
            numel_loc = a.data.numel()
            numel += numel_loc
            nnz += torch.count_nonzero(a.data).item()
    #
    return nnz/numel
    

def conv_sparsity(model):
    nnz = 0
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            s = m.weight.shape
            w = m.weight.view(s[0]*s[1], s[2]*s[3])
            nnz += torch.count_nonzero(torch.norm(w,p=1,dim=1)>0).item()
            total += s[0] * s[1]
    #
    if total == 0:
        return 0
    else:
        return nnz/total

def conv_effective_rank(model, epsilon=1e-3): # consider using fractional energy or stable rank
    """
    Computes the average effective rank of all Conv2d layers.
    The effective rank of a single layer is the count of singular values > epsilon.
    """
    total_rank = 0
    total_layers = 0

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            w = m.weight
            # Flatten each filter to one row: (out_channels, in_channels*kernel_height*kernel_width)
            mat = w.view(w.shape[0], -1)

            # SVD on the flattened matrix
            U, S, V = torch.svd(mat, some=True)

            # Count how many singular values exceed epsilon
            rank_layer = (S > epsilon).sum().item()
            total_rank += rank_layer
            total_layers += 1

    if total_layers == 0:
        return 0.0
    else:
        return total_rank / total_layers

def conv_effective_rank_ratio(model, epsilon=1e-6):
    """
    Computes the average ratio of effective rank to the maximum rank
    across all convolutional layers in the model. The effective rank
    is the count of singular values above epsilon. The maximum rank
    for a flattened weight matrix (m, n) is min(m, n).
    """
    total_ratio = 0.0
    total_layers = 0

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            w = m.weight
            # Flatten shape: (m, n) = (out_ch * in_ch, kernel_height * kernel_width)
            m_dim = w.size(0) * w.size(1)  # out_ch*in_ch
            n_dim = w.size(2) * w.size(3)  # kH*kW
            max_rank = min(m_dim, n_dim)

            mat = w.view(m_dim, n_dim)
            U, S, V = torch.svd(mat, some=True)  # economy SVD
            effective_rank = (S > epsilon).sum().item()

            # Ratio of effective rank to max rank
            layer_ratio = effective_rank / max_rank
            total_ratio += layer_ratio
            total_layers += 1

    if total_layers == 0:
        return 0.0
    else:
        return total_ratio / total_layers
    


def linear_effective_rank_ratio(model, epsilon=1e-6):
    """
    Computes the proportion of singular values above threshold epsilon
    for each fully connected layer, then averages these proportions.
    """
    layer_ratios = []

    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight
            U, S, V = torch.svd(w, some=True)
            
            # Proportion of singular values above threshold
            ratio = (S > epsilon).sum().item() / len(S)
            layer_ratios.append(ratio)
            
    if not layer_ratios:
        return 0.0
    else:
        return sum(layer_ratios) / len(layer_ratios)

def get_linear_layer_ranks(model, epsilon=1e-6):
    """
    Returns the effective rank ratio for each individual fully connected layer.
    """
    layer_ranks = {}
    
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            w = m.weight
            U, S, V = torch.svd(w, some=True)
            
            # Proportion of singular values above threshold
            ratio = (S > epsilon).sum().item() / len(S)
            layer_ranks[name] = ratio
            
    return layer_ranks

def get_conv_layer_ranks(model, epsilon=1e-6):
    """
    Returns the effective rank ratio for each individual convolutional layer.
    """
    layer_ranks = {}
    
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            w = m.weight
            # Reshape to 2D: (out_channels * in_channels, kernel_height * kernel_width)
            mat = w.view(w.size(0) * w.size(1), -1)
            
            # Compute SVD on the reshaped matrix
            U, S, V = torch.svd(mat, some=True)
            
            # Calculate maximum possible rank
            max_rank = min(mat.shape[0], mat.shape[1])
            
            # Proportion of singular values above threshold
            ratio = (S > epsilon).sum().item() / max_rank
            layer_ranks[name] = ratio
            
    return layer_ranks

def get_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            yield m.weight
        else:
            continue
            
def get_weights_conv(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            yield m.weight
        else:
            continue
            
def get_weights_linear(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            yield m.weight
        else:
            continue
            
def get_weights_batch(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            yield m.weight
        else:
            continue
            
def get_bias(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
            if not (m.bias is None):
                yield m.bias
        else:
            continue