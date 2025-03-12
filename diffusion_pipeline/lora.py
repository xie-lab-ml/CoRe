import torch
import torch.nn as nn

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, 
                 weak_lora_alpha=0.1, number_of_lora=1):
        super().__init__()
        self.linear = linear
        self.lora = nn.ModuleList([LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        ) for _ in range(number_of_lora)])
        self.use_lora = True
        self.lora_idx = 0
        
    def forward(self, x):
        if self.use_lora:
            return self.linear(x) + self.lora[self.lora_idx](x)
        else:
            return self.linear(x)

def replace_linear_with_lora(module, rank=64, alpha=1., tag=0, weak_lora_alpha=0.1, number_of_lora=1):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(child, rank, alpha, weak_lora_alpha=weak_lora_alpha, number_of_lora=number_of_lora))
        else:
            replace_linear_with_lora(child, rank, alpha, tag, weak_lora_alpha=weak_lora_alpha, number_of_lora=number_of_lora)


def lora_false(model, lora_idx=0):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            module.use_lora = False
            module.lora_idx = lora_idx

def lora_true(model, lora_idx=0):
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            module.use_lora = True
            module.lora_idx = lora_idx
            for i, lora in enumerate(module.lora):
                if i != lora_idx:
                    lora.A.requires_grad = False
                    lora.B.requires_grad = False
                    if lora.A.grad is not None:
                        del lora.A.grad
                    if lora.B.grad is not None:
                        del lora.B.grad
                else:
                    lora.A.requires_grad = True
                    lora.B.requires_grad = True
