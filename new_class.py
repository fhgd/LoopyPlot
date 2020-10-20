import inspect
from functools import wraps


def system(cls):
    names = {}
    for name, attr in cls.__dict__.items():
        if name.startswith('__'):
            continue
        try:
            func, defaults = attr
            names[name] = func
            for idx, (n, default) in enumerate(defaults.items()):
                if callable(names.get(n, None)):
                    continue
                names[n] = default
            setattr(cls, name, func)
        except TypeError:
            pass
    setattr(cls, '_names', names)
    return cls


def function(func):
    defaults = {}
    parameters = inspect.signature(func).parameters
    for n, param in parameters.items():
        if (param.kind is inspect.Parameter.VAR_POSITIONAL or
            param.kind is inspect.Parameter.VAR_KEYWORD
        ):
            continue
        if param.default is param.empty:
            defaults[n] = None
        else:
            defaults[n] = param.default
    #~ @wraps(func)
    def wrap(self):
        kwargs = {}
        for name in defaults:
            try:
                kwargs[name] = self._names[name]()
            except TypeError:
                kwargs[name] = self._names[name]
        return func(**kwargs)
    wrap.__name__ = func.__name__
    wrap.__doc__ = func.__doc__
    return wrap, defaults


def state(func=None, init=0):
    defaults = {func.__name__: init}
    def wrap(func):
        return StateValue(init, func), defaults

    if func is None:
        return wrap
    else:
        return wrap(func)


class StateValue:
    """minimal version from new.py"""
    def __init__(self, init, func, *args, **kwargs):
        self._init = init
        self(init)
        self._func = Func(func, self, *args, **kwargs)

    def __call__(self, value=None):
        if value is None:
            return self._value
        else:
            self._value = value
            self._is_initialized = True

    def next(self):
        self(self._func())
        return self()

    def reset(self):
        self(self._init)


def Func(func, *args, **kwargs):
    def eval():
        _args = []
        for v in args:
            try:
                _args.append(v())
            except TypeError:
                _args.append(v)
        _kwargs = {}
        for k, v in kwargs.items():
            try:
                _kwargs[k] = v()
            except TypeError:
                _kwargs[k] = v
        #~ print(f'{func.__name__}({_args}, {_kwargs})')
        return func(*_args, **_kwargs)
    return eval


@system
class Sweep:
    @function
    def sweep(idx=3, start=10, stop=20, step=2, num=None):
        if step is None:
            delta = stop - start
            step = delta / num
        return start + step * idx

    @state
    def idx(idx):
        return idx + 1

    def next(self):
        return self.idx.next()

x = Sweep()

#~ x = Sweep(10, 20, step=2)
#~ x()
#~ x.next()

