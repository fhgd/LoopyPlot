class _System:
    def __init__(self, func):
        self._ip = {}
        self._state = {}
        self._aux = {}
        self._op = {}
        self._func = func

    def input(self, func):
        self._ip[func.__name__] = func

    def state(self, func):
        self._state[func.__name__] = func

    def aux(self, func):
        self._aux[func.__name__] = func

    def output(self, func):
        self._op[func.__name__] = func

    def __call__(self):
        return self._op[self._func.__name__]()

    def _from_func(self, *args, **kwargs):
        self.func(sys, **kwargs)


##################


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
        return func(*_args, **_kwargs)
    return eval


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
            return value

    @property
    def _next(self):
        return self._func()

    def next(self):
        return self(self._next)

    def __repr__(self):
        return f'<state: {self()}>'


class Sys:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._state = {}
        self._func = None
        self._op = {}

    def _add_state(self, func, init):
        name = func.__name__
        obj = StateValue(init, func)
        self._state[name] = obj
        self.__setattr__(name, obj)
        return obj

    def state(self, func=None, init=0):
        def wrap(func):
            obj = self._add_state(func, init)
            return obj

        if func is None:
            return wrap
        else:
            return wrap(func)

    def returns(self, func):
        self._func = func
        return func

    def op(self, func):
        name = func.__name__
        obj = func
        self._op[name] = obj
        self.__setattr__(name, obj)
        return obj

    def __call__(self):
        args = tuple(state() for state in self._state.values())
        #~ return self._func(*args)
        args = args + self._args
        return self._func(*args, **self._kwargs)

    #~ def next(self):
        #~ states = self._state.values()
        #~ values = tuple(state._next for state in states)
        #~ for state, value in zip(states, values):
            #~ state(value)
        #~ if len(values) > 1:
            #~ return values
        #~ else:
            #~ return values[0]


def system(func):
    def system_factory(*args, **kwargs):
        sys = Sys(*args, **kwargs)
        func(sys)
        return sys
    return system_factory


if __name__ == '__main__':

    # ideas:
    #   * all system functions (and inputs) as properties
    #   * state: next property, update() function

    #~ @system
    def sweep(sys, start, stop, num=None, step=None):
        @sys.returns
        def sweep():
            delta = stop - start
            step_ = delta / num if step is None else step
            return start + step_ * idx

        @sys.state
        def idx(idx):
            return idx + 1

        @sys.op
        def next():
            return sys.idx.next()

    sys = Sys(10, 20, step=2)
    sweep(sys, 10, 20, step=2)

    # or

    @system
    def sweep(sys):
        @sys.state
        def idx(idx):
            return idx + 1

        @sys.op
        def next():
            return sys.idx.next()

        @sys.returns
        def sweep(idx, start, stop, num=None, step=None):
            delta = stop - start
            if step is None:
                step = delta / num
            return start + step*idx

    x = sweep(10, 20, step=2)
    x()
    x.next()


