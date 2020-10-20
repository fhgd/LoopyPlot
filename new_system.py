class Sys:
    def __init__(self):
        self._inp = {}
        self._out = {}
        self._aux = {}
        self._states = {}

    def state(self, func=None, init=0):
        if func is None:
            def wrap(func):
                self._states[func.__name__] = func
                return init
            return wrap
        else:
            self._states[func.__name__] = func
        return init

    def output(self, func):
        self._out[func.__name__] = func

    def aux(self, func):
        self._aux[func.__name__] = func

    def __call__(self):
        func = list(self._out.values())[0]
        return func()


#~ @System
def sweep(sys, start, stop, num=None, step=None):
    @sys.state
    def idx():
        return idx + 1

    @sys.output
    def value():
        delta = stop - start
        #~ _num = delta / step if num is None else num
        _step = delta / num if step is None else step
        return start + _step*idx

    @sys.output
    def is_finished():
        return value >= stop


sys = Sys()
sweep(sys, 0, 10, step=2)


if 0:

    class sweep(System):
        def config(sys, start, stop, num=None, step=None):
            @sys.state
            def idx():
                return idx + 1

            @sys.output
            def value():
                delta = stop - start
                #~ _num = delta / step if num is None else num
                _step = delta / num if step is None else step
                return start + _step*idx

            @sys.output
            def is_finished():
                return value >= stop


    class sweep(System):
        start = System.input()
        stop  = System.input()
        num   = System.input(None)
        step  = System.input(None)

        @System.state
        def idx():
            return idx + 1

        @System.output
        def value():
            delta = stop - start
            #~ _num = delta / step if num is None else num
            _step = delta / num if step is None else step
            return start + _step*idx

        @System.output
        def is_finished():
            return value >= stop


    class sweep(System):
        start = System.input()
        stop  = System.input()
        num   = System.input()
        step  = System.input()

        @sys.input_args
        def __init__(sys, start, stop, num=None, step=None):
            pass

        def __init__(sys, start, stop, num=None, step=None):
            sys.ip.start = start
            sys.ip.stop = stop
            sys.ip.num = num
            sys.ip.step = step

        def __init__(sys, start, stop, num=None, step=None):
            sys.add_input('start', start)
            sys.add_input('stop', stop)
            sys.add_input('num', num)
            sys.add_input('step', step)

            sys.add_output(myfunc, returns=['x', 'y', 't_0'])



        def idx(sys):
            return sys.state.idx + 1

        @sys.output
        def value(sys):
            delta = sys.ip.stop - sys.ip.start
            #~ _num = delta / step if num is None else num
            step = sys.ip.step
            if step is None:
                step = delta / sys.ip.num
            return sys.ip.start + step * sys.state.idx

        @sys.output
        def is_finished(sys):
            return sys.op.value >= sys.ip.stop


    @System.from_func(returns=['y1', 'y2'])
    def myfunc(gain, offs):
        return y1, y2

    @System.from_func(sys_arg='sys')
    def myfunc(sys, gain, offs):
        t = sys.ip.t
        sys.op.y1 = gain*t + offs
        sys.op.y2 = gain*t + offs



    # ala dataclasses

    class sweep(Sweep):
        start = System.input()

        idx = System.state(init=0)
        idx: state = 0.0

        stop: input

    @System(returns=System.output('op', ['z', '']))
    def myfunc(x, y):
        op.z = x + y


    def myfunc(x, y):
        op.z = x + y
    myfunc = System(myfunc, op=['z', '_t'])


    def myfunc(x, y):
        """
            op: z
                _t
        """
        op.z = x + y


    def myfunc(
        x,
        y,
    ) -> (
        'a'
    ):
        return x + y



    class sweep(System):
        start = System.input
        stop  = System.input
        num   = System.input(default=None)
        step  = System.input(default=None)

        idx = System.state(init=0)

        value       = System.output()
        is_finished = System.output()

        def func(sys):
            sys.op.value = sys.ip.start + 321

        @sys.state
        def idx():
            return idx + 1

        @sys.output
        def value():
            delta = stop - start
            #~ _num = delta / step if num is None else num
            _step = delta / num if step is None else step
            return start + _step*idx

        @sys.output
        def is_finished():
            return value >= stop


    sys = System(myfunc, returns=['value', 'is_finished'])


    class sweep(System):
        start = System.input
        stop  = System.input
        num   = System.input(default=None)
        step  = System.input(default=None)

        value       = System.output()
        is_finished = System.output()

        def func(sys):
            sys.op.value = sys.ip.start + 321



    # final ideals

    #~ @System
    def sweep(sys, start, stop, num=None, step=None):
        @sys.state
        def idx():
            return idx + 1

        @sys.output
        def value():
            delta = stop - start
            #~ _num = delta / step if num is None else num
            _step = delta / num if step is None else step
            return start + _step*idx

        @sys.output
        def is_finished():
            return value >= stop

        @sys.output('value')
        @sys.output('is_finished', 'is_running')
        def func():
            delta = stop - start
            _step = delta / num if step is None else step
            value = start + _step*idx

            is_finished = value >= stop
            is_running = not is_finished

            sys.op.is_finished = value >= stop
            sys.op.is_running = not is_finished



    #~ @System
    def measure(sys, acc, freq, RL):
        """
        Returns:
            absVpz
            argVpz
        """

        @sys.output('absVpz')
        @sys.output('argVpz')
        def func():
            delta = stop - start
            _step = delta / num if step is None else step
            value = start + _step*idx

            is_finished = value >= stop
            is_running = not is_finished

            sys.op.is_finished = value >= stop
            sys.op.is_running = not is_finished


    @System
    def sweep(start, stop, num=None, step=None):
        """
        Returns:
            value
            is_running
            is_finished
        """
        delta = stop - start
        _step = delta / num if step is None else step
        value = start + _step*idx

        is_finished = value >= stop
        is_running = not is_finished

        sys.op.is_finished = value >= stop
        sys.op.is_running = not is_finished


    class sweep(System):
        start = System.input
        stop  = System.input
        num   = System.input(default=None)
        step  = System.input(default=None)

        @System.state
        def idx(sys):
            return sys.idx + 1

        @System.output
        def value(sys):
            delta = sys.stop - sys.start
            #~ _num = delta / step if num is None else num
            _step = delta / sys.num if sys.step is None else sys.step
            return sys.start + _step * sys.idx

        @System.output('is_finished')
        @System.output('is_running')
        def _func(sys):
            return sys.value >= sys.stop

