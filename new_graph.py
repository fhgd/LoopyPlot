import bisect
import networkx as nx
import pandas as pd

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
        self._idx = 1  # == max(of all ixds) + 1

    def write(self, name, value, overwrite=False):
        idxs, values = self._data.setdefault(name, ([], []))
        if overwrite:
            idxs[-1:] = [self._idx]
            values[-1:] = [value]
        else:
            idxs.append(self._idx)
            values.append(value)
        self._idx += 1

    def read(self, name, idx=-1):
        idxs, values = self._data[name]
        idx_left = bisect.bisect_right(idxs, idx) - 1 if idx else idx
        # todo:  avoid bisect_right() - 1 = -1
        return values[idx_left]

    def last_idx(self, name):
        idxs, _ = self._data.get(name, ([0], None))
        return idxs[-1]

    def values(self, keys):
        idxs, _ = self._data[keys[-1]]
        return [[self.read(key, idx) for key in keys] for idx in idxs]

    def to_yaml(self, fname=''):
        pass

    @classmethod
    def from_yaml(self, fname=''):
        pass


class Node:
    __count__ = 0

    def __init__(self, name='', overwrite=False, lazy=True):
        # todo: underscore all attributes
        self._name = name
        self.overwrite = overwrite
        self.lazy = lazy
        self.id = Node.__count__
        Node.__count__ += 1
        self.key = f'n{self.id}'
        self._args = {}  # arg_name: arg_node
        # set by register(tm)
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
        return self

    def __call__(self):
        return self.eval()

    def eval(self):
        return self.tm.eval(self)

    #todo: add empty __eval__()

    def get(self):
        return self.tm.dm.read(self.key)

    def set(self, value):
        self.tm.dm.write(self.key, value, self.overwrite)
        return value

    @property
    def _last_idx(self):
        return self.tm.dm.last_idx(self.key)

    def __repr__(self):
        return f'n{self.id}_{self._name}' if self._name else f'n{self.id}'

    def _inputs(self):
        g = self.tm.g
        nodes = nx.shortest_path_length(g, target=self)
        #~ nodes = nx.single_source_shortest_path_length(g.reverse(copy=False), self)
        return {n: nodes[n] for n, d in g.in_degree(nodes) if d == 0}

    def _new_inputs(self):
        dm = self.tm.dm
        idx = self._last_idx
        inps = {}
        for n, d in self._inputs().items():
            if n._last_idx > idx:
                inps[n] = d
        return inps

    def _is_new(self, idx):
        return idx == 0 or self._last_idx > idx

    def _has_results(self):
        return self.key in self.tm.dm._data

    def _has_new_args(self):
        dm = self.tm.dm
        idx = self._last_idx
        if not idx:
            return True
        for arg_node in self._args.values():
            if arg_node._last_idx > idx:
                return True
        return False


class ValueNode(Node):
    def __init__(self, value=NOTHING, name=''):
        Node.__init__(self, name)
        self._value = value

    def register(self, tm):
        Node.register(self, tm)
        self.set(self._value)
        return self

class FuncNode(Node):
    def __init__(self, func, overwrite=False, lazy=True):
        Node.__init__(self, func.__name__, overwrite, lazy)
        self.func = func
        self.sweep = Nested()
        # todo: howto treat FuncNode args pointing to sweeps (SystemNode)?
        # todo: leave FuncNode as plain as possible, but is this a SystemNode?
        # todo: if yes, then move .sweep, .run(), .table() into SystemNode
        #
        # discussion: by
        #   func.arg.x1 = Sweep(10, 20, 0.2)
        #   func.arg.x2 = Sweep(0, 1, num=100)
        # this FuncNode 'needs' .is_running() and .next() from Nested.
        #
        # todo: How to combine FuncNode, Nested and all Sweeps?
        # todo: Is this a SystemNode with sub-systems? Sweeps are systems!
        # todo: Or should every node get a .is_running()?
        #       - could be usefull for Concat
        #
        # todo: How about RunFuncNode(FuncNode, SystemNode)?
        #       - increase the functionality by subclasses

    def run(self):
        self.tm.run(self)
        return self.table

    @property
    def table(self):
        names = []
        nodes = []
        for name, node in self._args.items():
            names.append(name)
            nodes.append(node)
        names.append(self._name)
        nodes.append(self)
        keys = [n.key for n in nodes]
        df = pd.DataFrame(self.tm.dm.values(keys))
        df.columns = names
        return df


class StateNode(Node):
    def __init__(self, name, init=0):
        Node.__init__(self, name)
        self._init = init
        self._next = FuncNode(lambda x: x, lazy=False)

    def register(self, tm):
        Node.register(self, tm)
        Node.register(self._next, tm)
        self._next.key = self.key
        self.reset()
        return self

    def reset(self):
        self.set(self._init)

    def next(self):
        return self._next.eval()

    def add_next(self, func, **kwargs):
        if self not in kwargs.values():
            kwargs[self._name] = self
        self._next._name = f'{self._name}_next'
        self._next.func = func
        self.tm._add_kwargs(self._next, kwargs)
        return self._next


class Container:
    def _add(self, node):
        name = node._name
        idx = 2
        while name in self.__dict__:
            name = f'{node._name}_{idx}'
            idx += 1
        setattr(self, name, node)


class TaskManager:
    def __init__(self):
        self.g = nx.DiGraph()
        self.dm = DataManager()
        self.func = Container()
        self.state = Container()

    def eval(self, node, lazy=None):
        _g = self.g.reverse(copy=False)
        nodes = nx.algorithms.dfs_postorder_nodes(_g, node)
        for n in nodes:
            if n.lazy and not n._has_new_args():
                continue
            kwargs = {name: node.get() for name, node in n._args.items()}
            retval = n.func(**kwargs)
            n.set(retval)
        return node.get()

    def run(self, fn):
        while 1:
            self.eval(fn)
            if fn.sweep.is_running():
                fn.sweep._next_state()
            else:
                break

    def add_state(self, name, init):
        node = StateNode(name, init)
        node.register(self)
        self.state._add(node)
        return node

    def add_func(self, func, **kwargs):
        node = FuncNode(func).register(self)
        self.func._add(node)
        self._add_kwargs(node, kwargs)
        return node

    def _add_kwargs(self, func_node, kwargs):
        for name, obj in kwargs.items():
            node = self._as_node(obj)
            if not node._name:
                node._name = f'{func_node._name}_arg_{name}'
            self.g.add_edge(node, func_node)
            func_node._args[name] = node

    def _as_node(self, obj):
        try:
            node = obj.register(self)
        except AttributeError:
            node = ValueNode(obj).register(self)
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
    # todo: try to replace Function with FuncNode
    def __init__(self, func):
        self._name = func.__name__
        self._func = func


class State:
    # todo: try to replace State with StateNode
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
    # todo: rename it into SystemNode and inherit from Node
    # todo: define uniqe system output as @Function def output(self): ...
    # todo: point .__eval__() to .output.__eval__()
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
        # todo: is class decorator more pythonic?
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
            node._name = f'arg_{name}'
            self._nodes[name] = node

        for fn in cls._functions:
            node = FuncNode(fn._func).register(tm)
            self._nodes[fn._name] = node
            _func_nodes.append(node)

        for state in cls._states:
            name = state._name
            node = StateNode(name, state._init)
            node._next.func = state._func
            node._next._name = f'{name}_next'
            node.register(tm)
            self._nodes[name] = node
            _func_nodes.append(node._next)

        for fnode in _func_nodes:
            params = inspect.signature(fnode.func).parameters
            for name, param in params.items():
                if (param.kind is inspect.Parameter.VAR_POSITIONAL or
                    param.kind is inspect.Parameter.VAR_KEYWORD):
                    continue
                if name not in self._nodes:
                    print(f'param {name!r} is not in nodes')
                    continue
                anode = self._nodes[name]
                tm.g.add_edge(anode, fnode)
                fnode._args[name] = anode
        return self

    def is_running(self):
        return 0 <= self.idx < len(self) - 1

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


class Sequence(BaseSweep):
    items = InP()

    @state(init=0)
    def idx(idx):
        return idx + 1

    @Function
    def value(idx, items):
        return items[idx]

    def __len__(self):
        return len(self.items)


class Nested:
    """Iterate over nested sweeps.

    >>> s1 = Sweep(10, 20, step=10).register(tm)
    >>> s2 = Sweep(1, 2, step=1).register(tm)
    >>> n = Nested(s1, s2)
    >>> n.as_list()
    [(10, 1), (10, 2), (20, 1), (20, 2)]
    """
    # todo: find a general strategy for FuncNode.next()
    #     Nested:
    #         * Zip: Concat(Sweep()) => offs  (normal sweep)
    #         * Zip: Concat(val, Sweep, val, Sweep) => gain  (iteration of values)
    #         * Zip:
    #             - Concat(Sweep(), value, Sweep(), value, ...) => x1  (zipped values)
    #             - Concat(Sweep(), value, Sweep(), value, ...) => x2

    # todo: arg-config api for sweeps
    #
    #   func.arg.log = True
    #   func.arg.offs = tm.Sweep(...)
    #   func.arg.gain = tm.Concat(val, val)
    #   func.arg.x1 = tm.Concat(tm.Sweep, val, tm.Sweep, val)
    #   func.arg.x2 = tm.Concat(tm.Sweep, val, tm.Sweep, val)
    #   func.arg.zip('x1, x2')
    #
    #   tm.Concat          iterate over all sweeps and values
    #   tm.ConcatHold      interrupt after each sub-sweep
    #   tm.ConcatIter      just iterater over the top-level items
    #
    #   func.arg.log = True
    #   func.arg.offs.Sweep(...)

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


class Zip:
    """Zip several sweeps together

    >>> s1 = Sweep(1, 3)
    >>> s2 = Iterate(10, 20, 30)
    >>> Zip(s1, s2).as_list()
    [(1, 10), (2, 20), (3, 30)]
    """
    def __init__(self, *sweeps):
        self.sweeps = list(sweeps)

    def __len__(self):
        return min(len(sweep) for sweep in self.sweeps)

    def reset(self):
        for sweep in self.sweeps:
            sweep.reset()

    def value(self):
        return tuple(s.value() for s in self.sweeps)

    def is_running(self):
        return all(s.is_running() for s in self.sweeps)

    def _next_state(self):
        for sweep in self.sweeps:
            sweep._next_state()
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

if 0:
    g = Sweep(100, 200, num=2).register(tm)
    d = Sweep(2, 5).register(tm)
    s = Sweep(0, 10, num=d.value).register(tm)


    n = Nested(g, d, s)
    # n.as_list()


if 1:
    def myfunc(x, gain, offs=0):
        print(f'gain = {gain}')
        print(f'   x = {x}')
        print(f'offs = {offs}')
        print()
        return gain*x + offs

    x = Sweep(10, 20, step=5).register(tm)
    g = Sequence([1, 10]).register(tm)

    tm.add_func(myfunc, x=x.value, gain=g.value, offs=0)

    tm.func.myfunc.sweep = Nested(g, x)

    z = Zip(x, g)

    df = tm.func.myfunc.run()
    #     x  gain  offs  myfunc
    # 0  10     1     0      10
    # 1  15     1     0      15
    # 2  20     1     0      20
    # 3  10    10     0     100
    # 4  15    10     0     150
    # 5  20    10     0     200


"""

def myfunc(a, b, c=123):
    return a + b + c


mytask = tm.add_func(myfunc, name='mytask', a=1, b=2)
tm.tasks.mytask.arg.c = tm.Sweep(5)
mytask.arg.c = tm.Sweep(5)

tm.state.go_from('INIT', 'OFF', tm.tasks.mytask)
tm.state.INIT.go_to('OFF', tm.tasks.mytast)



"""


### Use mutable data types for inplace state transitions whithout logging ###


if 0:
    def value(start, step, idx):
        idx[0] += 1
        return start + idx[0] * step

    tm.add_func(value, start=100, step=11, idx=[0])


if 0:
    # todo: idx node should have a flag for mutable datatype
    idx = ValueNode([0], name='idx')

    def value(start, step, idx):
        idx[0] += 1
        return start + idx[0] * step
    tm.add_func(value, start=100, step=11, idx=idx)
    tm.func.value.overwrite = True
    tm.func.value.lazy = False

    tm.func.value.eval()
    tm.func.value.eval()
    tm.func.value.eval()

    df = tm.func.value.table
    # start  step  idx  value
    # 0    100    11  [3]    133


if 0:
    @tm.add_func
    def new_idx():
        return [0]

    def value(start, step, idx):
        idx[0] += 1
        return start + idx[0] * step

    tm.add_func(value, start=100, step=11, idx=new_idx)
    #~ tm.func.value.overwrite = True
    tm.func.value.lazy = False

    tm.func.value.eval()
    tm.func.value.eval()
    tm.func.value.eval()

    df = tm.func.value.table
    # start  step  idx  value
    # 0    100    11  [3]    111
    # 1    100    11  [3]    122
    # 2    100    11  [3]    133

