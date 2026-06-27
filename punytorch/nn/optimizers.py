import numpy as np

from punytorch.tensor import Tensor


class Optimizer:
    """
    Base class for all optimizers.

    An optimizer is used to adjust the attributes of a neural network such as weights and learning rate in order to reduce the losses.
    """

    def __init__(self, params):
        """
        Args:
            params (iterable): An iterable of Parameters that define what to optimize.
        """
        self.params = list(params)

    @staticmethod
    def _as_array(value):
        if isinstance(value, Tensor):
            return value.data
        return np.asarray(value)

    @classmethod
    def _grad_array(cls, param):
        if param.grad is None:
            return None
        return cls._as_array(param.grad)

    def zero_grad(self):
        """
        Clears the gradients of all optimized Parameters.

        This is typically used when you start to compute gradients for the next optimization step.
        """
        for param in self.params:
            grad = self._grad_array(param)
            if grad is not None:
                param.grad = np.zeros_like(grad)

    def step(self):
        """
        Performs a single optimization step (parameter update).

        This method should be overridden by all subclasses, so we'll raise an error if it's not.

        Raises:
            NotImplementedError: If not overridden by subclasses.
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Implements the Stochastic Gradient Descent (SGD) optimization algorithm.

    SGD is a variant of gradient descent. Instead of performing computations on the whole dataset
    which is computationally expensive, SGD only computes on a small subset or a batch of the dataset.
    """

    def __init__(self, params, lr):
        """
        Args:
            params (iterable): An iterable of Parameters that define what to optimize.
            lr (float): The learning rate.
        """
        super().__init__(params)
        self.lr = lr

    def step(self):
        """
        Performs a single optimization step (parameter update).

        For each parameter in the list, the parameter's data is updated in-place by subtracting
        the learning rate times the parameter's gradient.
        """
        for param in self.params:
            grad = self._grad_array(param)
            if grad is None:
                continue
            param.data -= self.lr * grad


class Adam(Optimizer):
    """
    Implements the Adam optimization algorithm.

    Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent
    procedure to update network weights iteratively based on training data.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        """
        Args:
            params (iterable): An iterable of Parameters that define what to optimize.
            lr (float): The learning rate.
            betas (tuple of floats): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to improve numerical stability.

        Attributes:
            m (list): The first moment vector (the running average of the gradient).
            v (list): The second moment vector (the running average of the gradient squared).
            t (int): The timestep, used for bias correction.
        """
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]
        self.t = 0

    def step(self):
        """
        Performs a single optimization step (parameter update).

        For each parameter in the list, the parameter's data is updated in-place using the Adam update rule.
        """
        self.t += 1
        for i, param in enumerate(self.params):
            grad_data = self._grad_array(param)
            if grad_data is None:
                continue
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad_data
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad_data**2)
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    """
    Implements the RMSProp optimization algorithm.

    RMSProp is an optimizer that utilizes the magnitude of recent gradients to normalize the gradients.
    We usually combine this with a momentum term.
    """

    def __init__(self, params, lr, alpha=0.99, eps=1e-8):
        """
        Args:
            params (iterable): An iterable of Parameters that define what to optimize.
            lr (float): The learning rate.
            alpha (float): Smoothing constant. A smaller value results in a smaller moving average and a larger value results in a longer moving average.
            eps (float): Term added to improve numerical stability.

        Attributes:
            v (list): The second moment vector (the running average of the gradient squared).
        """
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        """
        Performs a single optimization step (parameter update).

        For each parameter in the list, the parameter's data is updated in-place using the RMSProp update rule.
        """
        for i, param in enumerate(self.params):
            grad = self._grad_array(param)
            if grad is None:
                continue
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad**2)
            param.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)


class Adagrad(Optimizer):
    """
    Implements the Adagrad optimization algorithm.

    Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative to
    how frequently a parameter gets updated during training. The more updates a parameter receives,
    the smaller the learning rate.
    """

    def __init__(self, params, lr, eps=1e-8):
        """
        Args:
            params (iterable): An iterable of Parameters that define what to optimize.
            lr (float): The learning rate.
            eps (float): Term added to improve numerical stability.

        Attributes:
            G (list): A list that stores the sum of the squares of the gradients for each parameter.
        """
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.G = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        """
        Performs a single optimization step (parameter update).

        For each parameter in the list, the parameter's data is updated in-place using the Adagrad update rule.
        """
        for i, param in enumerate(self.params):
            grad = self._grad_array(param)
            if grad is None:
                continue
            self.G[i] += grad**2
            param.data -= self.lr / np.sqrt(self.G[i] + self.eps) * grad


class Adadelta(Optimizer):
    """
    Implements the Adadelta optimization algorithm.

    Adadelta is an extension of Adagrad that seeks to reduce its aggressive, monotonically decreasing learning rate.
    Instead of accumulating all past squared gradients, Adadelta restricts the window of accumulated past gradients to some fixed size w.
    """

    def __init__(self, params, rho=0.9, eps=1e-6):
        """
        Args:
            params (iterable): An iterable of Parameters that define what to optimize.
            rho (float): Coefficient used for decaying average of squared gradients.
            eps (float): Term added to improve numerical stability.

        Attributes:
            Eg (list): A list that stores the decaying average of past squared gradients.
            Edelta (list): A list that stores the decaying average of past squared updates.
        """
        super().__init__(params)
        self.rho = rho
        self.eps = eps
        self.Eg = [np.zeros_like(param.data) for param in self.params]
        self.Edelta = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        """
        Performs a single optimization step (parameter update).

        For each parameter in the list, the parameter's data is updated in-place using the Adadelta update rule.
        """
        for i, param in enumerate(self.params):
            grad = self._grad_array(param)
            if grad is None:
                continue
            self.Eg[i] = self.rho * self.Eg[i] + (1 - self.rho) * grad**2
            delta = np.sqrt((self.Edelta[i] + self.eps) / (self.Eg[i] + self.eps)) * grad
            self.Edelta[i] = self.rho * self.Edelta[i] + (1 - self.rho) * delta**2
            param.data -= delta


class Adamax(Optimizer):
    """
    Implements the Adamax optimization algorithm.

    Adamax is a variant of Adam based on the infinity norm. It is less sensitive to learning rate and provides more stable and robust updates.
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        """
        Args:
            params (iterable): An iterable of Parameters that define what to optimize.
            lr (float): The learning rate.
            betas (tuple of floats): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to improve numerical stability.

        Attributes:
            m (list): The first moment vector (the running average of the gradient).
            u (list): The infinity norm (maximum absolute value) of the gradient.
            t (int): The timestep, used for bias correction.
        """
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.u = [np.zeros_like(param.data) for param in self.params]
        self.t = 0

    def step(self):
        """
        Performs a single optimization step (parameter update).

        For each parameter in the list, the parameter's data is updated in-place using the Adamax update rule.
        """
        self.t += 1
        for i, param in enumerate(self.params):
            grad = self._grad_array(param)
            if grad is None:
                continue
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            self.u[i] = np.maximum(self.betas[1] * self.u[i], np.abs(grad))
            param.data -= (self.lr / (1 - self.betas[0] ** self.t)) * self.m[i] / (self.u[i] + self.eps)
