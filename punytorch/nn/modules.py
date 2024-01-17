import numpy as np

from punytorch.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, tensor):
        super().__init__(tensor, requires_grad=True)


class Module:
    """
    Base class for all neural network modules.

    All models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes.
    """

    def __call__(self, *args, **kwargs) -> Tensor:
        """
        Makes the instance callable like a function and directs the arguments to the forward method.

        Args:
            *args, **kwargs: Variable length argument list and keyword arguments to be passed to the forward method.

        Returns:
            Tensor: The output of the forward method.
        """
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Parameter]:
        """
        Returns a list of all parameters in the module.

        Returns:
            list[Parameter]: A list of all parameters in the module.
        """
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
        """
        Returns a dictionary containing a whole state of the module.

        Returns:
            dict: A dictionary containing a whole state of the module.
        """

        def absorb_dict(root: dict, d: dict):
            for k, v in d.items():
                root[k] = v

        def _get_params(root: Module, prefix=""):
            d = {}
            for k, v in root.__dict__.items():
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
        """
        Copies parameters and buffers from state_dict into this module and its descendants.

        Args:
            state_dict (dict): A dict containing parameters and persistent buffers.
        """
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
        """
        Adds a persistent buffer to the module.

        Args:
            name (str): Name of the buffer. The buffer can be accessed from this module using the given name.
            value (Tensor): Tensor to be registered.
        """
        value = value.clone().detach() if isinstance(value, Tensor) else value
        setattr(self, name, value)

    def to(self, device):
        """
        Moves and/or casts the parameters and buffers.

        Args:
            device: The device to move the parameters and buffers to.

        Returns:
            Module: self
        """
        return self  # add gpu backend maybe

    def eval(self):
        """
        Sets the module in evaluation mode.
        """
        for p in self.parameters():
            p.requires_grad = False

    def train(self):
        """
        Sets the module in training mode.
        """
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, *args, **kwargs):
        """
        Defines the computation performed at every call.

        Should be overridden by all subclasses.

        Args:
            *args, **kwargs: Variable length argument list and keyword arguments.

        Raises:
            NotImplementedError: If not overridden by subclasses.
        """
        raise NotImplemented


class Linear(Module):
    """
    Implements a linear transformation of the input data: y = xA^T + b

    This module can be used as a layer in a neural network model.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initializes the Linear module.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True
        """
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            np.random.rand(out_features, in_features)
            / np.sqrt(in_features + out_features)
        )
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.

        Returns:
            Tensor: The output of the linear transformation.
        """
        self.input = x
        x = x @ self.weight.T
        if self.bias:
            x = x + self.bias
        return x

    def backward(self, grad: Tensor) -> Tensor:
        """
        Computes the backward pass of the Linear module for backpropagation.

        Args:
            grad (Tensor): The gradient of the loss with respect to the output of this module.

        Returns:
            Tensor: The gradient of the loss with respect to the input of this module.
        """
        # Compute the gradient with respect to the input
        grad_input = grad @ self.weight.data

        # Compute the gradient with respect to the weights
        grad_weight = self.input.T @ grad

        # Compute the gradient with respect to the bias
        grad_bias = np.sum(grad, axis=0)

        # Store the gradients in the .grad attributes of the weights and bias
        self.weight.grad = grad_weight
        if self.bias is not None:
            self.bias.grad = grad_bias

        return grad_input


class ModuleList(Module, list):
    """
    Holds submodules in a list.

    ModuleList can be indexed like a regular Python list, but modules it contains
    are properly registered, and will be visible by all Module methods.
    """

    def __init__(self, modules=None):
        """
        Initializes the ModuleList.

        Args:
            modules (iterable, optional): An iterable of modules to add.
        """
        super().__init__()
        if modules is not None:
            self += modules

    def append(self, module: Module):
        """
        Appends a module to the list.

        Args:
            module (Module): Module to append.
        """
        super().append(module)

    def __setitem__(self, i: int, module: Module):
        """
        Sets the module at the given index.

        Args:
            i (int): Index to set the module at.
            module (Module): Module to set.
        """
        return super().__setitem__(i, module)

    def parameters(self):
        """
        Returns a list of all parameters in the module list.

        Returns:
            list[Parameter]: A list of all parameters in the module list.
        """
        params = []
        for module in self:
            params.extend(module.parameters())
        return params
