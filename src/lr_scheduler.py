import torch
from torch.optim.lr_scheduler import _LRScheduler
# Modified from https://github.com/cmpark0126/pytorch-polynomial-lr-decay

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, total_num_step, end_learning_rate=0.0, power=1.0, warmup_updates=0):
        if total_num_step <= 1.:
            raise ValueError('total_num_step should be greater than 1.')
        self.total_num_step = total_num_step
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        self.warmup_updates = warmup_updates
        if warmup_updates > 0:
            self.warmup_factor = 1.0 / warmup_updates
        else:
            self.warmup_factor = 1
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.total_num_step:
            return [self.end_learning_rate for _ in self.base_lrs]
        elif self.warmup_updates > 0 and self.last_step < self.warmup_updates:
            self.warmup_factor = self.last_step / float(self.warmup_updates)
            return [self.warmup_factor*base_lr for base_lr in self.base_lrs]
        else:
            return [(base_lr - self.end_learning_rate) * 
                    ((1 - (self.last_step-self.warmup_updates) / (self.total_num_step-self.warmup_updates)) ** (self.power)) + 
                    self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step > self.total_num_step:
            decay_lrs = [self.end_learning_rate for _ in self.base_lrs]
        elif self.warmup_updates > 0 and self.last_step < self.warmup_updates:
            self.warmup_factor = self.last_step / float(self.warmup_updates)
            decay_lrs = [self.warmup_factor*base_lr for base_lr in self.base_lrs] 
        else:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                        ((1 - (self.last_step-self.warmup_updates) / (self.total_num_step-self.warmup_updates)) ** (self.power)) + 
                        self.end_learning_rate for base_lr in self.base_lrs]
            
        for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
            param_group['lr'] = lr

class InverseSqrtLRDecay(_LRScheduler):
    """
    InverseSqrt learning rate decay
    """
    
    def __init__(self, optimizer, warmup_updates=0):
        self.last_step = 0
        self.warmup_updates = warmup_updates
        if warmup_updates > 0:
            self.warmup_factor = 1.0 / warmup_updates
        else:
            self.warmup_factor = 1
        # self.decay_factor = warmup_updates ** 0.5
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.warmup_updates > 0 and self.last_step < self.warmup_updates:
            self.warmup_factor = self.last_step / float(self.warmup_updates)
            return [self.warmup_factor*base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr * self.warmup_updates ** 0.5 * self.last_step ** -0.5 for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.warmup_updates > 0 and self.last_step < self.warmup_updates:
            self.warmup_factor = self.last_step / float(self.warmup_updates)
            decay_lrs = [self.warmup_factor*base_lr for base_lr in self.base_lrs] 
        else:
            decay_lrs = [base_lr * self.warmup_updates ** 0.5 * self.last_step ** -0.5 for base_lr in self.base_lrs]
            
        for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
            param_group['lr'] = lr


if __name__ == '__main__':
    v = torch.zeros(10)
    optim = torch.optim.SGD([v], lr=0.01)
    scheduler = PolynomialLRDecay(optim, total_num_step=1000, end_learning_rate=1e-8, power=1.0,warmup_updates=100)
    a = []
    for epoch in range(1, 1500):
        scheduler.step()
        a.append(optim.param_groups[0]['lr'])
    print(a)
        
    
