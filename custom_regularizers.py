import torch
import regularizers as reg

class DynamicRankNuclearRegularizer(reg.reg_nuclear_linear):
    """Nuclear norm regulariser with dynamically adjustable rank constraint"""
    
    def __init__(self, lamda=1.0, initial_rank=16, max_rank=None):
        super().__init__(lamda=lamda)
        self.initial_rank = initial_rank
        self.current_rank = initial_rank
        self.max_rank = max_rank
        
    def _svd(self, x):
        min_dim = min(x.shape)
        
        if self.current_rank is not None and self.current_rank < min_dim:
            try:
                U, S, V = torch.svd_lowrank(x, q=self.current_rank, niter=2)
            except RuntimeError:
                U, S, V = torch.svd(x, some=True)
        else:
            U, S, V = torch.svd(x, some=True)
            
        return U, S, V
        
    def __call__(self, x):
        """Calculate nuclear norm with current rank"""
        U, S, V = self._svd(x)
        return self.lamda * torch.sum(S)
        
    def prox(self, x, delta=1.0):
        """Apply proximal operator with current rank constraint"""
        U, S, V = self._svd(x)
        S_thresh = torch.clamp(S - self.lamda * delta, min=0.0)
        return (U * S_thresh.unsqueeze(0)) @ V.t()
        
    def set_rank(self, new_rank):
        """Update the rank constraint"""
        if self.max_rank is not None:
            new_rank = min(new_rank, self.max_rank)
        self.current_rank = new_rank
        return self.current_rank
        
    def get_rank(self):
        """Get current rank constraint"""
        return self.current_rank