import bisect
import networkx as nx

import inspect
from types import FunctionType, SimpleNamespace


class _Nothing:
    """
    Sentinel class to indicate the lack of a value when ``None`` is ambiguous.

    ``_Nothing`` is a singleton. There is only ever one of it.

    from attr/_make.py
    """

    _singleton = None

    def __new__(cls):
        if _Nothing._singleton is None:
            _Nothing._singleton = super(_Nothing, cls).__new__(cls)
        return _Nothing._singleton

    def __repr__(self):
        return "NOTHING"


NOTHING = _Nothing()


class DataManager:
    def __init__(self):
        self._data = {}
        self._idx = 0  # == max(of all ixds) + 1

    def write(self, name, value, idx=0):
        idxs, values = self._data.setdefault(name, ([], []))
        idxs.append(self._idx)
        values.append(value)
        self._idx += 1

    def read(self, name, idx=-1):
        idxs, values = self._data[name]
        idx_left = bisect.bisect_right(idxs, idx) - 1 if idx else idx
        # todo:  avoid bisect_right() - 1 = -1
        return values[idx_left]

    def to_yaml(self, fname=''):
        pass

    @classmethod
    def from_yaml(self, fname=''):
        pass


class Node:
    __count__ = 0

    def __init__(self, name=''):
        self.name = name
        self.id = Node.__count__
        Node.__count__ += 1
        self.key = f'n{self.id}'
        self._tm = None

    @property
    def tm(self):
        if self._tm is None:
            msg = f'node {self!r} must registered to a tm instance'
            raise ValueError(msg)
        return self._tm

    def register(self, tm):
        tm.g.add_node(self)
        self._tm = tm

    def __call__(self):
        return self.eval()

    def eval(self):
        return self.tm.eval(self)

    def get(self):
        return self.tm.dm.read(self.key)

    def set(self, value):
        self.tm.dm.write(self.key, value)
        return value

    def __repr__(self):
        return f'n{self.id}_{self.name}' if self.name else f'n{self.id}'


class ValueNode(Node):
    def __init__(self, value=NOTHING, name=''):
        Node.__init__(self, name)
        self._value = value

    def register(self, tm):
        Node.register(self, tm)
        self.set(self._value)


class FuncNode(Node):
    def __init__(self, func):
        Node.__init__(self, func.__name__)
        self.func = func


class StateNode(Node):
    def __init__(self, name, init=0):
        Node.__init__(self, name)
        self._init = init
        self._next = FuncNode(lambda x: x)
        self._is_initialized = True

    def register(self, tm):
        Node.register(self, tm)
        tm.g.add_node(self._next)
        self._next._tm = tm
        self._next.key = self.key
        self.reset()

    def set_value(self, value):
        self.set(value)
        self._is_initialized = True

    def reset(self):
        self.set_value(self._init)

    def next(self):
        if self._is_initialized:
            self._is_initialized = False
            return self._next.get()
        else:
            return self._next.eval()

    def add_next(self, func, **kwargs):
        if self not in kwargs.values():
            kwargs[self.name] = self
        self._next.name = f'{self.name}_next'
        self._next.func = func
        self.tm._add_kwargs(self._next, kwargs)
        return self._next


class Container:
    def _add(self, node):
        name = node.name
        idx = 2
        while name in self.__dict__:
            name = f'{node.name}_{idx}'
            idx += 1
        setattr(self, name, node)


class TaskManager:
    def __init__(self):
        self.g = nx.DiGraph()
        self.dm = DataManager()
        self.func = Container()
        self.state = Container()

    def eval(self, node):
        dct = nx.shortest_path_length(self.g, target=node)
        nodes = sorted(dct, key=dct.get, reverse=True)
        for n in nodes:
            if not isinstance(n, FuncNode):
                continue
            kwargs = {}
            for edge in self.g.in_edges(n):
                name = self.g.edges[edge]['arg']
                kwargs[name] = edge[0].get()
            retval = n.func(**kwargs)
            n.set(retval)
        return node.get()

    def add_state(self, name, init):
        node = StateNode(name, init)
        node.register(self)
        self.state._add(node)
        return node

    def add_func(self, func, **kwargs):
        node = FuncNode(func)
        node.register(self)
        self.func._add(node)
        self._add_kwargs(node, kwargs)
        return node

    def _add_kwargs(self, func_node, kwargs):
        for name, obj in kwargs.items():
            node = self._as_node(obj)
            if not node.name:
                node.name = f'{func_node.name}_arg_{name}'
            self.g.add_edge(node, func_node, arg=name)

    def _as_node(self, obj):
        if hasattr(obj, 'id'):
            return obj
        else:
            node = ValueNode(obj)
            node.register(self)
            return node

    def sweep(self, start, stop, step=1, num=None):
        return Sweep(start, stop, step, num).register(self)


tm = TaskManager()

def quad(x, gain=1, offs=0):
    return gain * x**2 + offs

def double(x):
    return 2*x

idx = tm.add_state('idx', 0)
idx.add_next(lambda x: x + 1, x=idx)
#~ idx.add_next(lambda idx: idx + 1)


"""
idx = tm.add_state('idx', init=0)

@idx.add_next(x=idx)
def func(x):
    return x + 1


@tm.new_state
def idx(idx=0, x=other_node_func, y=123):
    # use only kwargs if next_func should be configured completely
    return idx + 1


@tm.new_func
def myquad(x, y=idx, z=123, gain=Sweep(5)):
    return x + y + z

myquad.arg.x = tm.sweep(5)
myquad.arg.x.sweep(5)

"""

tm.add_func(double, x=idx)
tm.add_func(quad, x=2, gain=tm.func.double, offs=0)

print(tm.eval(tm.func.quad))
for n in range(4):
    tm.eval(idx._next)
    print(tm.eval(tm.func.quad))

tm.dm._data


"""
@tm.new_func
def sweep(start, stop, step=1, num=None, idx=0):
    return start + step*delta

s = sweep(1, 5, 0.5)
s.eval()
s()

# or

s.value
s.next
s.is_running
"""


class InP:
    cls_counter = 0

    def __init__(self, default=NOTHING):
        InP.cls_counter += 1
        self._counter = InP.cls_counter
        self._default = default
        self._name = ''


class Function:
    def __init__(self, func):
        self._name = func.__name__
        self._func = func


class State:
    def __init__(self, name, init, func):
        self._name = name
        self._init = init
        self._func = func


def state(func=None, init=0):
    def wrap(func):
        return State(func.__name__, init, func)
    if func is None:
        return wrap
    else:
        return wrap(func)


class SysBase:
    _inputs = []
    _functins = []
    _states = []


class BaseSweep:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._nodes = {}
        self._tm = None

    def __repr__(self):
        clsname = self.__class__.__name__
        args = [repr(value) for value in self._args]
        for name, value in self._kwargs.items():
            args.append(f'{name}={value!r}')
        args = ', '.join(args)
        return f'{clsname}({args})'

    def register(self, tm):
        cls = self.__class__
        try:
            cls._inputs
        except AttributeError:
            cls._inputs = []
        try:
            cls._functions
        except AttributeError:
            cls._functions = []
        try:
            cls._states
        except AttributeError:
            cls._states = []

        _inputs = []
        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, InP):
                _inputs.append(attr)
                attr._name = name
                def node_getter(self, name=name):
                    return self._nodes[name].eval()
                setattr(cls, name, property(node_getter))
            elif isinstance(attr, Function):
                cls._functions.append(attr)
                def node_getter(self, name=name):
                    return self._nodes[name]
                setattr(cls, name, property(node_getter))
            elif isinstance(attr, State):
                cls._states.append(attr)
                def node_getter(self, name=name):
                    return self._nodes[name].get()
                setattr(cls, name, property(node_getter))

        for inp in sorted(_inputs, key=lambda inp: inp._counter):
            cls._inputs.append(inp)

        # init of instance with nodes
        self._tm = tm
        self._nodes = {}
        _func_nodes = []

        for idx, inp in enumerate(cls._inputs):
            name = inp._name
            try:
                value = self._args[idx]
            except IndexError:
                value = self._kwargs.get(name, inp._default)
            if value is NOTHING:
                msg = f'provide value for input argument {name!r}'
                raise ValueError(msg)
            node = tm._as_node(value)
            node.name = f'arg_{name}'
            self._nodes[name] = node

        for fn in cls._functions:
            node = FuncNode(fn._func)
            node.register(tm)
            self._nodes[fn._name] = node
            _func_nodes.append(node)

        for state in cls._states:
            name = state._name
            node = StateNode(name, state._init)
            node._next.func = state._func
            node._next.name = f'{name}_next'
            node.register(tm)
            self._nodes[name] = node
            _func_nodes.append(node._next)

        for node in _func_nodes:
            params = inspect.signature(node.func).parameters
            for name, param in params.items():
                if (param.kind is inspect.Parameter.VAR_POSITIONAL or
                    param.kind is inspect.Parameter.VAR_KEYWORD):
                    continue
                if name not in self._nodes:
                    print(f'param {name!r} is not in nodes')
                    continue
                tm.g.add_edge(self._nodes[name], node, arg=name)
        return self

    def is_running(self):
        return 0 <= self.idx < len(self) - 1

    def is_finished(self):
        return not self.is_running()

    def reset(self):
        nodes = self._nodes
        for state in self._states:
            nodes[state._name].reset()

    def _next_state(self, name=''):
        names = [name] if name else [s._name for s in self._states]
        nodes = self._nodes
        return tuple(nodes[name]._next.eval() for name in names)

    def next(self):
        if self.is_running():
            if self._is_initialized:
                self._is_initialized = False
            else:
                self._next_state()
        else:
            raise StopIteration
        return self.value()

    __next__ = next

    def __iter__(self):
        return self

    def as_list(self):
        self.reset()
        return list(self)

    @property
    def _is_initialized(self):
        names = [s._name for s in self._states]
        nodes = self._nodes
        return any(nodes[name]._is_initialized for name in names)

    @_is_initialized.setter
    def _is_initialized(self, value):
        nodes = self._nodes
        for s in self._states:
            nodes[s._name]._is_initialized = value


class Sweep(BaseSweep):
    start = InP()
    stop  = InP()
    num   = InP(None)
    step  = InP(1)

    @Function
    def aux(start, stop, step, num):
        delta = stop - start
        if num is None:
            _step = abs(step) if delta > 0 else -abs(step)
            _num = int(delta / _step) + 1
        else:
            div = num - 1
            div = div if div > 1 else 1
            _step = float(delta) / div
            _num = num
        return SimpleNamespace(num=_num, step=_step)

    @state(init=0)
    def idx(idx):
        return idx + 1

    @Function
    def value(idx, start, aux):
        return start + aux.step * idx

    def __len__(self):
        return self.aux().num

    @property
    def min(self):
        return min(self.start, self.stop)

    @property
    def max(self):
        return max(self.start, self.stop)


class Nested:
    """Iterate over nested sweeps.

    >>> s1 = Sweep(1, 2)
    >>> s2 = Iterate(10, 20)
    >>> n = Nested(s1, s2)
    >>> n.as_list()
    [(1, 10), (2, 10), (1, 20), (2, 20)]
    """
    def __init__(self, *sweeps):
        self.sweeps = list(sweeps)
        self._is_initialized = True

    def __len__(self):
        value = 1
        for sweep in self.sweeps:
            value *= len(sweep)
        return value

    def reset(self):
        for sweep in self.sweeps:
            sweep.reset()
        self._is_initialized = True

    def value(self):
        return tuple(s.value() for s in self.sweeps)

    def is_running(self):
        return any(s.is_running() for s in self.sweeps)

    def is_finished(self):
        return not self.is_running()

    def _next_state(self):
        sweeps = self.sweeps

        n = 0
        while sweeps[n].is_finished():
            n += 1
        sweeps[n]._next_state()

        n -= len(sweeps)
        while n > -len(sweeps):
            n -= 1
            sweep = sweeps[n]
            if sweep.is_finished():
                sweep.reset()
            elif sweep._is_initialized:
                print(f'is init: {sweep._is_initialized}')
                sweep._next_state()
        return self.value()

    def next(self):
        if self.is_running():
            if self._is_initialized:
                self._is_initialized = False
            else:
                self._next_state()
        else:
            raise StopIteration
        return self.value()

    __next__ = next

    def __iter__(self):
        return self

    def as_list(self):
        self.reset()
        return list(self)


s = Sweep(5, 20, num=idx).register(tm)
d = Sweep(1, 2).register(tm)
g = Sweep(100, 200, num=2).register(tm)


n = Nested(s, d, g)
# n.as_list()

"""

def myfunc(a, b, c=123):
    return a + b + c


mytask = tm.add_func(myfunc, name='mytask', a=1, b=2)
tm.tasks.mytask.arg.c = tm.Sweep(5)
mytask.arg.c = tm.Sweep(5)

tm.state.go_from('INIT', 'OFF', tm.tasks.mytask)
tm.state.INIT.go_to('OFF', tm.tasks.mytast)



"""