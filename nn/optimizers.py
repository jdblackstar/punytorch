import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad


class Adam(Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]
        self.t = 0

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.grad)

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (
                param.grad**2
            )
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp(Optimizer):
    def __init__(self, params, lr, alpha=0.99, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.v = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (param.grad**2)
            param.data -= self.lr * param.grad / (np.sqrt(self.v[i]) + self.eps)


class Adagrad(Optimizer):
    def __init__(self, params, lr, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.G = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.G[i] += param.grad**2
            param.data -= self.lr / np.sqrt(self.G[i] + self.eps) * param.grad


class Adadelta(Optimizer):
    def __init__(self, params, rho=0.9, eps=1e-6):
        super().__init__(params)
        self.rho = rho
        self.eps = eps
        self.Eg = [np.zeros_like(param.data) for param in self.params]
        self.Edelta = [np.zeros_like(param.data) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.Eg[i] = self.rho * self.Eg[i] + (1 - self.rho) * param.grad**2
            delta = (
                np.sqrt((self.Edelta[i] + self.eps) / (self.Eg[i] + self.eps))
                * param.grad
            )
            self.Edelta[i] = self.rho * self.Edelta[i] + (1 - self.rho) * delta**2
            param.data -= delta


class Adamax(Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.u = [np.zeros_like(param.data) for param in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
            self.u[i] = np.maximum(self.betas[1] * self.u[i], np.abs(param.grad))
            param.data -= (
                (self.lr / (1 - self.betas[0] ** self.t))
                * self.m[i]
                / (self.u[i] + self.eps)
            )
