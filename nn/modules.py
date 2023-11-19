import numpy as np
from punytorch.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor, requires_grad=True)

class Module:
    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Parameter]:
        params = []
        for key, value in self.__dict__.items():
            if isinstance(value, Parameter):
                params.append(value)
            if isinstance(value, Module):
                params.extend(value.parameters())
            if isinstance(value, ModuleList):
                for module in value:
                    params.extend(module.parameters())
        return list(set(params))

    def state_dict(self):
        def absorb_dict(root: dict, d: dict):
            for k, v in d.items():
                root[k] = v

        def _get_params(root: Module, prefix=""):
            d = {}
            for k, v in root.__dict__.items():
                print(k)
                if isinstance(v, Parameter):
                    key = f"{prefix}.{k}" if prefix != "" else f"{k}"
                    d[key] = v.clone()
                elif isinstance(v, ModuleList):
                    ds = [
                        _get_params(
                            m, f"{prefix}.{k}.{idx}" if prefix != "" else f"{k}.{idx}"
                        )
                        for idx, m in enumerate(v)
                    ]
                    for x in ds:
                        absorb_dict(d, x)
                elif isinstance(v, Module):
                    absorb_dict(
                        d, _get_params(v, f"{prefix}.{k}" if prefix != "" else f"{k}")
                    )
            return d

        return _get_params(self)

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            attr = getattr(self, key, None)
            if attr is None:
                raise KeyError(f"Key {key} not found in module's state.")
            if isinstance(attr, Parameter):
                attr.data = value
            elif isinstance(attr, Module):
                attr.load_state_dict(value)
            elif isinstance(attr, list):  # Assuming ModuleList
                for param, state in zip(attr, value):
                    param.load_state_dict(state)

    def register_buffer(self, name, value: Tensor):
        value = value.clone().detach() if isinstance(value, Tensor) else value
        setattr(self, name, value)

    def to(self, device):
        return self  # add gpu backend maybe
    
    def eval(self):
        for p in self.parameters():
            p.requires_grad = False
        
    def train(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, *args, **kwargs):
        raise NotImplemented


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            np.rand((out_features, in_features))  / np.sqrt(in_features+out_features)
        )
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        x = x @ self.weight.t()
        if self.bias:
            x = x + self.bias
        return x
    
class ModuleList(Module, list):
    def __init__(self, modules=None):
        super().__init__()
        if modules is not None:
            self += modules

    def append(self, module: Module):
        super().append(module)

    def __setitem__(self, i: int, module: Module):
        return super().__setitem__(i, module)

    def parameters(self):
        params = []
        for module in self:
            params.extend(module.parameters())
        return params