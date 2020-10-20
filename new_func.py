def Value(value):
    def eval():
        return value
    return eval


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

def function(func):
    def wrap(*args, **kwargs):
        return Func(func, *args, **kwargs)
    return wrap



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

    #~ def __repr__(self):
        #~ return str(self())


def state(func=None, init=0):
    def wrap(func):
        return StateValue(init, func)

    if func is None:
        return wrap
    else:
        return wrap(func)


if __name__ == '__main__':

    @state
    def idx(idx):
        return idx + 1

    @function
    def sweep(idx, start, stop, num=None, step=None):
        delta = stop - start
        if step is None:
            step = delta / num
        return start + step*idx

    x = sweep(idx, 10, 20, step=2)
    # besser:
    # x = sweep(10, 20, step=2)
    # x()
    # x.next()

    @function
    def mystep(x):
        return 2*x

    @function
    def sequence(idx, *sequence):
        try:
            return sequence[idx]()
        except TypeError:
            return sequence[idx]

    @function
    def myquad(x, gain=1, offs=0):
        return gain*x**2 + offs

    step = mystep(1)
    gain = sequence(idx, step, 2, 5, 10)
    m = myquad(x, gain, offs=1)
    m()


    # oder:
    # task = myquad()
    # task.args.x = sweep(10, 20, step=2)
    # task.args.gain = sequence(step, 2, 5, 10)
    # task.args.offs = 1

    # task()
    # task.run(idx=-1)
