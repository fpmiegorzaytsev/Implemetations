import torch
from torch import Tensor

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.99), weight_decay=0.01):
        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('sparse not allowed')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                state['step'] += 1

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                extra_exp_avg = beta1 * exp_avg + (1 - beta1) * grad

                step_vec = group['lr'] * (torch.sign(extra_exp_avg) + group['weight_decay'] * p.data)
                exp_avg.mul_(beta2).add_(1 - beta2, grad)

                p.data.sub_(step_vec)

        return loss
