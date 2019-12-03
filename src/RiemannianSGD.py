from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class RiemannianSGDHyperparameter(tpe.Protocol):
        lr = None  # type: float


_default_hyperparam = optimizer.Hyperparameter()  # type: RiemannianSGDHyperparameter # NOQA
_default_hyperparam.lr = 0.01


class RiemannianSGDRule(optimizer.UpdateRule):

    """Update rule of vanilla stochastic gradient descent.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        lr (float): Learning rate.

    """
    _kernel = None

    def __init__(self, parent_hyperparam=None, lr=None):
        super(RiemannianSGDRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr

    def update_core_cpu(self, param):
        grad = param.grad
        data = param.data
        if grad is None:
            return
        if isinstance(param.data, intel64.mdarray):
            param.data.inplace_axpby(1.0, -self.hyperparam.lr, grad)
        else:
            param.data -= self.hyperparam.lr * grad


class RiemannianSGD(optimizer.GradientMethod):

    """Vanilla Stochastic Gradient Descent.

    Args:
        lr (float): Learning rate.

    """

    def __init__(self, lr=_default_hyperparam.lr):
        super(RiemannianSGD, self).__init__()
        self.hyperparam.lr = lr

    lr = optimizer.HyperparameterProxy('lr')

    def create_update_rule(self):
        return RiemannianSGDRule(self.hyperparam)
