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

    def _last_idx(self, name):
        idxs, _ = self._data.get(name, ([0], None))
        return idxs[-1]

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
        if self._new_inputs():
            return self.eval()
        else:
            return self.get()

    def eval(self):
        return self.tm.eval(self)

    def get(self):
        return self.tm.dm.read(self.key)

    def set(self, value):
        self.tm.dm.write(self.key, value)
        return value

    def __repr__(self):
        return f'n{self.id}_{self.name}' if self.name else f'n{self.id}'

    def _inputs(self):
        g = self.tm.g
        nodes = nx.shortest_path_length(g, target=self)
        #~ nodes = nx.single_source_shortest_path_length(g.reverse(copy=False), self)
        return {n: nodes[n] for n, d in g.in_degree(nodes) if d == 0}

    def _new_inputs(self):
        dm = self.tm.dm
        idx = dm._last_idx(self.key)
        inps = {}
        for n, d in self._inputs().items():
            if dm._last_idx(n.key) > idx:
                inps[n] = d
        return inps


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

    def register(self, tm):
        Node.register(self, tm)
        tm.g.add_node(self._next)
        self._next._tm = tm
        self._next.key = self.key
        self.reset()

    def reset(self):
        self.set(self._init)

    def next(self):
        return self.tm.eval(self._next, lazy=False)
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

    def eval(self, node, lazy=True):
        dct = nx.shortest_path_length(self.g, target=node)
        nodes = sorted(dct, key=dct.get, reverse=True)
        _g = self.g.reverse(copy=False)
        nodes = nx.algorithms.dfs_postorder_nodes(_g, node)
        new_nodes = set(node._new_inputs())
        print(f'new: {new_nodes}')
        for n in nodes:
            if not isinstance(n, FuncNode):
                continue
            arg_nodes = {a for a, b in self.g.in_edges(n)}
            print(f'args of {n}: {arg_nodes}')
            if lazy and not arg_nodes.intersection(new_nodes):
                continue
            kwargs = {}
            for edge in self.g.in_edges(n):
                name = self.g.edges[edge]['arg']
                kwargs[name] = edge[0].get()
            retval = n.func(**kwargs)
            new_nodes.add(n)
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


if 0:
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

if 0:
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
                    return self._nodes[name].__call__()
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
        return tuple(nodes[name].next() for name in names)

    def next(self):
        new_inputs = self._nodes['value']._new_inputs()
        if not new_inputs:
            if self.is_running():
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

    >>> s1 = Sweep(10, 20, step=10).register(tm)
    >>> s2 = Sweep(1, 2, step=1).register(tm)
    >>> n = Nested(s1, s2)
    >>> n.as_list()
    [(10, 1), (10, 2), (20, 1), (20, 2)]
    """
    def __init__(self, *sweeps):
        self.sweeps = list(sweeps)

    def __len__(self):
        value = 1
        for sweep in self.sweeps:
            value *= len(sweep)
        return value

    def reset(self):
        for sweep in self.sweeps:
            sweep.reset()

    def value(self):
        return tuple(s.value() for s in self.sweeps)

    def is_running(self):
        return any(s.is_running() for s in self.sweeps)

    def is_finished(self):
        return not self.is_running()

    def _next_state(self):
        sweeps = []
        for sweep in reversed(self.sweeps):
            if sweep.is_running():
                sweep._next_state()
                break
            else:
                sweeps.append(sweep)
        for sweep in sweeps:
            sweep.reset()
        return [s.idx for s in self.sweeps]

    def next(self):
        new_inputs = []
        for sweep in self.sweeps:
            new_inputs += sweep._nodes['value']._new_inputs()
        if not new_inputs:
            if self.is_running():
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


if 0:
    g = Sweep(100, 200, num=2).register(tm)
    d = Sweep(2, 2).register(tm)
    s = Sweep(5, 15, num=3).register(tm)


    n = Nested(g, d, s)
    # n.as_list()

if 1:
    g = Sweep(100, 200, num=2).register(tm)
    d = Sweep(2, 5).register(tm)
    s = Sweep(0, 10, num=d.value).register(tm)


    n = Nested(g, d, s)
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
