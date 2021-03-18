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



"""
8.3.2021

@system
class Sweep:
    start = input()
    stop  = input()
    step  = input(1)
    num   = input(None)

    @state
    def idx(idx):
        return idx + 1

    def value(idx, start, aux):
        return start + aux.step * idx

    def is_finished(value, stop):
        return value >= stop

    def is_running(idx, aux):
        return idx < aux.num

    def len(aux):
        return aux.num

    def len(num: aux):
        return num

    def len(num=tm.extract('aux')):
        return num

    @returns('num, step')
    def aux(start, stop, step, num):
        '''Returns: num, step'''
        if num is None:
            delta = stop - start
            _step = abs(step) if delta > 0 else -abs(step)
            _num = int(delta / _step) + 1
        else:
            div = num - 1
            div = div if div > 1 else 1
            _step = float(delta) / div
            _num = num
        return _num, _step
        return SimpleNameSpace(num=_num, step=_step)
        return dict(num=_num, step=_step)



    def len(num: aux):
    def len(num: aux[num]):
    def len(num: aux.autoget):
    def len(num: aux.aget):
    def len(num: aux.partof):
    def len(num: aux.acon):

    def len(num: 'aux'):
    def len(num: 'aux.num'):

    def len(num=aux.num):
    def len(num=autoget('aux')):
    def len(num=partof('aux')):
    def len(num=aget('aux', 'num')):

    def len(num=autoconnect('aux', 'num')):

    ### ===== ###

    def len(num=acon('aux')):

    ### ----- ###

    def len(num=GET(aux)):
    def len(num=elem(aux)):
    def len(num=elem('aux')):


Vorschlag:
    kwargs nur für echte inputs, values sind ip-defaults, bei Konflikten
    eine Warnung erzeugen.

    Verkünpfungen zwischen Funktionen mittels annotations.
    kwargs haben dann keinen Einfluss.


"""
