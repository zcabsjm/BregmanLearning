import torch
import math
import regularizers as reg

class LinBreg(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,reg=reg.reg_none(), delta=1.0, momentum=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        defaults = dict(lr=lr, reg=reg, delta=delta, momentum=momentum)
        super(LinBreg, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            delta = group['delta']
            # define regularizer for this group
            reg = group['reg'] 
            step_size = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                # get grad and state
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # get prox
                    # initialize subgradients
                    state['sub_grad'] = self.initialize_sub_grad(p,reg, delta)
                    state['momentum_buffer'] = None
                # -------------------------------------------------------------
                # update scheme
                # -------------------------------------------------------------
                # get the current sub gradient
                sub_grad = state['sub_grad']
                # update on the subgradient
                if momentum > 0.0: # with momentum
                    mom_buff = state['momentum_buffer']
                    if state['momentum_buffer'] is None:
                        mom_buff = torch.zeros_like(grad)
 
                    mom_buff.mul_(momentum)
                    mom_buff.add_((1-momentum)*step_size*grad) 
                    state['momentum_buffer'] = mom_buff
                    #update subgrad
                    sub_grad.add_(-mom_buff)
                                                            
                else: # no momentum
                    sub_grad.add_(-step_size * grad)
                # update step for parameters
                p.data = reg.prox(delta * sub_grad, delta)
        
    def initialize_sub_grad(self,p, reg, delta):
        p_init = p.data.clone()
        return 1/delta * p_init + reg.sub_grad(p_init)
    
    @torch.no_grad()
    def evaluate_reg(self):
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            delta = group['delta']
            
            # define regularizer for this group
            reg = group['reg']
            
            # evaluate the reguarizer for each parametr in group
            for p in group['params']:
                group_reg_val += reg(p)
                
            # append the group reg val
            reg_vals.append(group_reg_val)
            
        return reg_vals
                
        
    
# ------------------------------------------------------------------------------------------------------    
class ProxSGD(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,reg=reg.reg_none()):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        defaults = dict(lr=lr, reg=reg)
        super(ProxSGD, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            # define regularizer for this group
            reg = group['reg'] 
            step_size = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                # get grad and state
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    
                # -------------------------------------------------------------
                # update scheme
                # -------------------------------------------------------------               
                # gradient steps
                p.data.add_(-step_size * grad)
                # proximal step
                p.data = reg.prox(p.data, step_size)
                
    @torch.no_grad()
    def evaluate_reg(self):
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            # define regularizer for this group
            reg = group['reg']
            
            # evaluate the reguarizer for each parametr in group
            for p in group['params']:
                group_reg_val += reg(p)
                
            # append the group reg val
            reg_vals.append(group_reg_val)
            
        return reg_vals
                   
# ------------------------------------------------------------------------------------------------------           
class AdaBreg(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,reg=reg.reg_none(), delta=1.0, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
            
        defaults = dict(lr=lr, reg=reg, delta=delta, betas=betas, eps=eps)
        super(AdaBreg, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            delta = group['delta']
            # get regularizer for this group
            reg = group['reg']
            # get parameters for adam
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                # get grad and state
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # get prox
                    # initialize subgradients
                    state['sub_grad'] = self.initialize_sub_grad(p,reg, delta)
                    state['exp_avg'] = torch.zeros_like(state['sub_grad'])
                    state['exp_avg_sq'] = torch.zeros_like(state['sub_grad'])
                # -------------------------------------------------------------
                # update scheme
                # -------------------------------------------------------------
                # update step
                state['step'] += 1
                step = state['step']
                # get the current sub gradient and averages
                sub_grad = state['sub_grad']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                
                # define bias correction factors
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # denominator in the fraction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                
                # step size in adam update
                step_size = lr / bias_correction1
                
                # update subgrad
                sub_grad.addcdiv_(exp_avg, denom, value=-step_size)
                
                # update step for parameters
                p.data = reg.prox(delta * sub_grad, delta)
        
    def initialize_sub_grad(self,p, reg, delta):
        p_init = p.data.clone()
        return 1/delta * p_init + reg.sub_grad(p_init)
    
    @torch.no_grad()
    def evaluate_reg(self):
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            delta = group['delta']
            
            # define regularizer for this group
            reg = group['reg']
            
            # evaluate the reguarizer for each parametr in group
            for p in group['params']:
                group_reg_val += reg(p)
                
            # append the group reg val
            reg_vals.append(group_reg_val)
            
        return reg_vals
    
    
class lamda_scheduler:
    '''scheduler for the regularisation parameter'''
    def __init__(self, opt,idx, warmup = 0, increment = 0.05, cooldown=0, target_sparse=1.0, reg_param ="mu"):
        self.opt = opt
        self.group = opt.param_groups[idx]
        
        # warm up
        self.warmup = warmup
        
        # increment
        self.increment = increment
        
        # cooldown
        self.cooldown_val = cooldown
        self.cooldown = cooldown
        
        # target
        self.target_sparse = target_sparse
        self.reg_param = reg_param
         
    def __call__(self, sparse, verbosity = 1):
        # check if we are still in the warm up phase
        if self.warmup > 0:
            self.warmup -= 1
        elif self.warmup == 0:
            self.warmup = -1
        else:
            # cooldown 
            if self.cooldown_val > 0:
                self.cooldown_val -= 1 
            else: # cooldown is over, time to update and reset cooldown
                self.cooldown_val = self.cooldown

                # discrepancy principle for lamda
                if sparse > self.target_sparse:
                    self.group['reg'].mu += self.increment
                else:
                    self.group['reg'].mu = max(self.group['reg'].mu - self.increment,0.0)
                
                # reset subgradients
                for p in self.group['params']:   
                    state = self.opt.state[p]
                    state['sub_grad'] = self.opt.initialize_sub_grad(p, self.group['reg'],  self.group['delta'])
                    
        if verbosity > 0:
            print('Lamda was set to:', self.group['reg'].mu, ', cooldown on:',self.cooldown_val)
            
class NuclearLambdaScheduler:
    """
    Decreases the lambda parameter for a regularizer in a specific parameter group
    when a monitored metric has stopped improving. This is intended for nuclear
    norm regularization to allow rank to increase when performance plateaus.

    Args:
        optimizer: Wrapped optimizer.
        group_idx (int): Index of the parameter group whose regularizer's lambda to schedule.
        patience (int): Number of epochs with no improvement after which lambda is reduced.
                        Default: 5.
        factor (float): Factor by which the lambda is reduced. new_lambda = lambda * factor.
                        Default: 0.5.
        min_lambda (float): Lower bound on the lambda parameter. Default: 1e-8.
        mode (str): One of `min` or `max`. In `min` mode, lambda will be reduced when the
                    quantity monitored has stopped decreasing; in `max` mode it will be reduced
                    when the quantity monitored has stopped increasing. Default: 'min'.
        verbose (bool): If True, prints a message to stdout for each update. Default: True.
    """
    def __init__(self, optimizer, group_idx, patience=5, factor=0.5, min_lambda=1e-8, mode='min', verbose=True):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        if mode not in {'min', 'max'}:
            raise ValueError('Mode must be min or max.')

        self.optimizer = optimizer
        self.group_idx = group_idx
        self.patience = patience
        self.factor = factor
        self.min_lambda = min_lambda
        self.mode = mode
        self.verbose = verbose

        try:
            self.reg_instance = self.optimizer.param_groups[self.group_idx]['reg']
            if not hasattr(self.reg_instance, 'lamda'):
                 raise AttributeError(f"Regularizer in group {group_idx} does not have a 'lamda' attribute.")
            self.current_lambda = float(self.reg_instance.lamda) 
        except IndexError:
            raise IndexError(f"Optimizer does not have parameter group with index {group_idx}.")
        except AttributeError as e:
             raise AttributeError(f"Could not access 'lamda' in regularizer for group {group_idx}: {e}")


        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.wait = 0
        self.update_scheduled = False 

    def step(self, metric_value):
        """ Call this after the validation phase, passing the metric to monitor. """
        improved = False
        if self.mode == 'min':
            if metric_value < self.best_metric:
                self.best_metric = metric_value
                improved = True
        else: # mode == 'max'
            if metric_value > self.best_metric:
                self.best_metric = metric_value
                improved = True

        if improved:
            self.wait = 0
            self.update_scheduled = False # Reset schedule if improved
        else:
            self.wait += 1

        if self.wait >= self.patience:
             # Only schedule update if not already scheduled in this stagnation period
            if not self.update_scheduled:
                old_lambda = self.current_lambda
                new_lambda = max(self.current_lambda * self.factor, self.min_lambda)
                if new_lambda < old_lambda:
                    # Schedule the update
                    self.reg_instance.lamda = new_lambda
                    self.current_lambda = new_lambda
                    self.update_scheduled = True # Mark update as scheduled
                    if self.verbose:
                        print(f"\nNuclearLambdaScheduler: Metric stagnated for {self.patience} epochs. Reducing lambda for group {self.group_idx} to {new_lambda:.2e}")
                elif self.verbose and not self.update_scheduled:
                     print(f"\nNuclearLambdaScheduler: Metric stagnated, but lambda already at minimum {self.min_lambda:.2e}.")
        # Reset wait counter if an update was just applied
        if self.update_scheduled and improved:
             self.wait = 0
             self.update_scheduled = False


class RankScheduler:
    """Scheduler for dynamically increasing the rank constraint"""
    
    def __init__(self, optimizer, group_idx, patience=2, increase_by=16, 
                 steps_until_max=5, mode='min', verbose=False):
        self.optimizer = optimizer
        self.group_idx = group_idx
        self.patience = patience
        self.increase_by = increase_by
        self.steps_until_max = steps_until_max
        self.mode = mode
        self.verbose = verbose
        
        # Check if the reg has dynamic rank capability
        if not hasattr(optimizer.param_groups[group_idx]['reg'], 'set_rank'):
            raise ValueError("Regularizer must support dynamic rank adjustment")
        
        self.best_value = float('inf') if mode == 'min' else -float('inf')
        self.wait_count = 0
        self.step_count = 0
        
    def step(self, metric_value):
        """Update rank when validation loss plateaus"""
        is_better = (metric_value < self.best_value) if self.mode == 'min' else (metric_value > self.best_value)
        
        if is_better:
            self.best_value = metric_value
            self.wait_count = 0
        else:
            self.wait_count += 1
            
        if self.wait_count >= self.patience and self.step_count < self.steps_until_max:
            # Increase rank
            reg = self.optimizer.param_groups[self.group_idx]['reg']
            current_rank = reg.get_rank()
            new_rank = current_rank + self.increase_by
            actual_new_rank = reg.set_rank(new_rank)
            
            if self.verbose:
                print(f"Increasing rank from {current_rank} to {actual_new_rank}")
                
            self.wait_count = 0
            self.step_count += 1


class LinBregNesterov(torch.optim.Optimizer):
    """LinBreg optimizer with Nesterov-like momentum acceleration.
    
    This implementation adapts Nesterov accelerated gradient to the Bregman learning framework.
    It approximates the effect of looking ahead without requiring additional gradient evaluations.
    
    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float): Learning rate (default: 1e-3)
        reg (object): Regularizer object (default: reg.reg_none())
        delta (float): Scaling parameter for proximal operator (default: 1.0)
        momentum (float): Momentum factor (default: 0.0)
    """
    
    def __init__(self, params, lr=1e-3, reg=reg.reg_none(), delta=1.0, momentum=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate")
        defaults = dict(lr=lr, reg=reg, delta=delta, momentum=momentum)
        super(LinBregNesterov, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            delta = group['delta']
            reg = group['reg']
            step_size = group['lr']
            momentum = group['momentum']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['sub_grad'] = self.initialize_sub_grad(p, reg, delta)
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                sub_grad = state['sub_grad']
                mom_buff = state['momentum_buffer']
                
                if momentum > 0:
                    # Store old momentum buffer for Nesterov update
                    old_mom_buff = mom_buff.clone()
                    
                    # Update momentum buffer (standard)
                    mom_buff.mul_(momentum).add_(step_size * grad)
                    
                    # Apply Nesterov correction
                    nesterov_update = (1 + momentum) * mom_buff - momentum * old_mom_buff
                    
                    # Update subgradient with Nesterov correction
                    sub_grad.add_(-nesterov_update)
                    
                    # Store updated momentum buffer
                    state['momentum_buffer'] = mom_buff
                else:
                    # Standard update without momentum
                    sub_grad.add_(-step_size * grad)
                
                # Apply proximal update
                p.data = reg.prox(delta * sub_grad, delta)
    
    def initialize_sub_grad(self, p, reg, delta):
        p_init = p.data.clone()
        return 1/delta * p_init + reg.sub_grad(p_init)
    
    @torch.no_grad()
    def evaluate_reg(self):
        reg_vals = []
        for group in self.param_groups:
            group_reg_val = 0.0
            delta = group['delta']
            reg = group['reg']
            for p in group['params']:
                group_reg_val += reg(p)
            reg_vals.append(group_reg_val)
        return reg_vals

