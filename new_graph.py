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

    def __contains__(self, node):
        return node._key in self._data

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
        self._overwrite = overwrite
        self._lazy = lazy
        self._id = Node.__count__
        Node.__count__ += 1
        self._key = f'n{self._id}'
        self._args = {}  # arg_name: arg_node
        self._root = None
        # set by _register(tm)
        self.__tm = None

    @property
    def _tm(self):
        if self.__tm is None:
            msg = f'node {self!r} must registered to a tm instance'
            raise ValueError(msg)
        return self.__tm

    def _register(self, tm):
        tm.g.add_node(self)
        self.__tm = tm
        return self

    @classmethod
    def _as_node(cls, obj):
        return obj if isinstance(obj, cls) else ValueNode(obj)

    def __call__(self):
        return self._eval()

    def _eval(self):
        return self._tm.eval(self)

    #todo: add empty __eval__()

    def _get(self):
        return self._tm.dm.read(self._key)

    def _set(self, value):
        self._tm.dm.write(self._key, value, self._overwrite)
        return value

    @property
    def _last_idx(self):
        return self._tm.dm.last_idx(self._key)

    def __repr__(self):
        return f'n{self._id}_{self._name}' if self._name else f'n{self._id}'

    def _inputs(self):
        g = self._tm.g
        nodes = nx.shortest_path_length(g, target=self)
        #~ nodes = nx.single_source_shortest_path_length(g.reverse(copy=False), self)
        return {n: nodes[n] for n, d in g.in_degree(nodes) if d == 0}

    def _new_inputs(self):
        dm = self._tm.dm
        idx = self._last_idx
        inps = {}
        for n, d in self._inputs().items():
            if n._last_idx > idx:
                inps[n] = d
        return inps

    def _is_new(self, idx):
        return idx == 0 or self._last_idx > idx

    def _has_results(self):
        return self._key in self._tm.dm._data

    def _has_new_args(self):
        dm = self._tm.dm
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

    def _register(self, tm):
        Node._register(self, tm)
        self._set(self._value)
        return self


class FuncNode(Node):
    def __init__(self, func=None, name='', overwrite=False, lazy=True):
        name = name if name or not func else func.__name__
        Node.__init__(self, name, overwrite, lazy)
        self._sweep = Nested()
        self._func = func
        # todo: howto treat FuncNode args pointing to sweeps (SystemNode)?
        # todo: leave FuncNode as plain as possible, but is this a SystemNode?
        # todo: if yes, then move ._sweep, .run(), .table() into SystemNode
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
        self._tm.run(self)
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
        keys = [n._key for n in nodes]
        df = pd.DataFrame(self._tm.dm.values(keys))
        df.columns = names
        return df


class StateNode(Node):
    def __init__(self, name, init=0):
        Node.__init__(self, name)
        self._init = ValueNode(init)
        self._init._root = self
        self._next = FuncNode(lambda x: x, overwrite=True)
        self._next._root = self

    def _register(self, tm):
        Node._register(self, tm)
        Node._register(self._next, tm)
        self._init._register(tm)
        self.reset()
        return self

    def reset(self):
        init = self._init._get()
        if self not in self._tm.dm or self._get() != init:
            self._set(init)

    def next_eval(self):
        return self._next._eval()

    def next_update(self):
        return self._set(self._next._get())

    def add_next(self, func, **kwargs):
        if self not in kwargs.values():
            kwargs[self._name] = self
        self._next._name = f'{self._name}_next'
        self._next._func = func
        self._tm._add_kwargs(self._next, kwargs)
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
            if n._lazy and not n._has_new_args():
                continue
            kwargs = {name: node._get() for name, node in n._args.items()}
            retval = n._func(**kwargs)
            n._set(retval)
        return node._get()

    def run(self, fn):
        while 1:
            self.eval(fn)
            if fn._sweep.is_running():
                fn._sweep._next_state()
            else:
                break

    def add_state(self, name, init):
        node = StateNode(name, init)
        node._register(self)
        self.state._add(node)
        return node

    def add_func(self, func, **kwargs):
        node = FuncNode(func)._register(self)
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
            node = obj._register(self)
        except AttributeError:
            node = ValueNode(obj)._register(self)
        return node

    def sweep(self, start, stop, step=1, num=None):
        return Sweep(start, stop, step, num)._register(self)


class InP:
    cls_counter = 0

    def __init__(self, default=NOTHING):
        InP.cls_counter += 1
        self._counter = InP.cls_counter
        self._default = default
        self._name = ''

    def __set_name__(self, cls, name):
        self._name = name

    def __get__(self, obj, cls=None):
        return obj._nodes.get(self._name, self._default)._eval()


class Function:
    def __init__(self, func, name=''):
        self._name = name if name else func.__name__
        self._func = func

    def __get__(self, obj, cls=None):
        return obj._nodes.get(self._name, self._func)


class State:
    def __init__(self, name, init, func):
        self._name = name
        self._init = init
        self._func = func

    def __get__(self, obj, cls=None):
        return obj._nodes.get(self._name, self._init)._get()


def state(func=None, init=0):
    def wrap(func):
        return State(func.__name__, init, func)
    if func is None:
        return wrap
    else:
        return wrap(func)


def zip_lazy(*args):
    iters = [iter(arg) for arg in args]
    while 1:
        values = []
        for it in iters:
            try:
                values.append(next(it))
            except StopIteration:
                pass
        if not values:
            break
        yield tuple(values)


class SystemNode(FuncNode):
    def __init__(self, *args, **kwargs):
        self._nodes = {}
        self._states = []
        self._subsys = []
        self._func_nodes = [self]

        inputs = []
        retval = Function(lambda: None, '__return__')

        for name, attr in self.__class__.__dict__.items():
            if isinstance(attr, InP):
                inputs.append(attr)
            elif isinstance(attr, Function):
                if attr._name == '__return__':
                    retval = attr
                else:
                    node = FuncNode(attr._func)
                    self._nodes[attr._name] = node
                    self._func_nodes.append(node)
            elif isinstance(attr, State):
                name = attr._name
                node = StateNode(name, attr._init)
                node._next._func = attr._func
                node._next._name = f'{name}_next'
                self._nodes[name] = node
                self._func_nodes.append(node._next)
                self._states.append(node)

        inputs = sorted(inputs, key=lambda inp: inp._counter)
        for idx, inp in enumerate(inputs):
            name = inp._name
            default = inp._default
            try:
                value = args[idx]
            except IndexError:
                value = kwargs.get(name, default)
            if value is NOTHING:
                msg = f'provide value for input argument {name!r}'
                raise ValueError(msg)
            node = Node._as_node(value)
            node._name = f'arg_{name}'
            self._nodes[name] = node

        argitems = [repr(value) for value in args]
        argitems += [f'{name}={val!r}' for name, val in kwargs.items()]
        name = f'{self.__class__.__name__}({", ".join(argitems)})'
        super().__init__(retval._func, name)

    def add_subsys(self, system, name='', **kwargs):
        self._subsys.append(system)
        return system

    def _register(self, tm):
        super()._register(tm)
        for node in self._nodes.values():
            node._register(tm)
        # todo: put this into __init__ as much as possible
        # todo: func_node._register(tm) should be enough!
        for fnode in self._func_nodes:
            params = inspect.signature(fnode._func).parameters
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

    def _show(self):
        for name, node in self._nodes.items():
            print(f'{name}:  {node}._value = {node._value}')

    def _next_state(self, name=''):
        states = tuple(self._iter_all_states())
        for state in states:
            state.next_eval()
        for state in states:
            state.next_update()

    def _iter_all_states(self):
        for state in self._states:
            yield state
        for subs in self._subsys:
            yield from subs._iter_all_states()

    def _next_updates(self):
        next_nodes = [ [set(state._next for state in self._states)] ]
        next_nodes += [sub._next_updates() for sub in self._subsys]
        return [set().union(*vals) for vals in zip_lazy(*next_nodes)]

    def update(self):
        for nodes in self._next_updates():
            #~ print(f'nodes for update: {nodes}')
            for node in nodes:
                node._eval()
            for node in nodes:
                node._root._set(node._get())
        return self._eval()


class BaseLoopNode(SystemNode):
    def _next(self):
        new_inputs = self._new_inputs()
        if not new_inputs:
            if self.is_running():
                self._next_state()
            else:
                raise StopIteration
        return self._eval()

    __next__ = _next

    def __iter__(self):
        return self

    def is_running(self):
        return 0 <= self.idx < len(self) - 1

    def reset(self):
        for state in self._states:
            state.reset()

    def as_list(self):
        self.reset()
        return list(self)


class Sweep(BaseLoopNode):
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
    def __return__(idx, start, aux):
        return start + aux.step * idx

    def __len__(self):
        return self.aux().num

    @property
    def min(self):
        return min(self.start, self.stop)

    @property
    def max(self):
        return max(self.start, self.stop)


class Sequence(BaseLoopNode):
    items = InP()

    @state(init=0)
    def idx(idx):
        return idx + 1

    @Function
    def __return__(idx, items):
        return items[idx]

    def __len__(self):
        return len(self.items)


class Nested:
    """Iterate over nested sweeps.

    >>> s1 = Sweep(10, 20, step=10)._register(tm)
    >>> s2 = Sweep(1, 2, step=1)._register(tm)
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
    #
    # todo: arg-config api for sweeps
    #
    #   func.arg.log = True
    #   func.arg.offs = tm.Sweep(...)
    #   func.arg.gain = tm.Concat(val, val)
    #   func.arg.x1 = tm.Concat(tm.Sweep, val, tm.Sweep, val)
    #   func.arg.x2 = tm.Concat(tm.Sweep, val, tm.Sweep, val)
    #   func.arg.zip('x1, x2')
    #
    #   # Basic Loops
    #   tm.Loop(val, ...)                        like Sequence(val, ...)
    #   tm.LoopLin(start, stop, step, num=None)  like Sweep()
    #   tm.LoopLog(start, stop, num=None)        like LogSweep
    #
    #   # Some Loop Operations
    #   tm.Concat          iterate over all sweeps and values
    #   tm.ConcatHold      interrupt after each sub-sweep
    #   tm.ConcatIter      just iterater over the top-level items
    #
    #   func.arg.log = True
    #   func.arg.offs.Sweep(...)
    #   func.arg(
    #       offs=Sweep(...),
    #       gain=123,
    #       x1=Loop(...),
    #   )

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
        return tuple(s._eval() for s in self.sweeps)

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
            new_inputs += sweep._new_inputs()
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


### Tests ###


tm = TaskManager()


if 1:
    def quad(x, gain=1, offs=0):
        return gain * x**2 + offs

    def double(x):
        return 2*x

    idx = tm.add_state('idx', 0)
    idx.add_next(lambda x: x + 1, x=idx)
    #~ idx.add_next(lambda idx: idx + 1)

    lst = Sequence([3, 4, 5])._register(tm)

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
s._eval()
s()

# or

s.value
s.next
s.is_running
"""


if 0:
    g = Sweep(100, 200, num=2)._register(tm)
    d = Sweep(2, 2)._register(tm)
    s = Sweep(5, 15, num=3)._register(tm)


    n = Nested(g, d, s)
    # n.as_list()

if 0:
    g = Sweep(100, 200, num=2)._register(tm)
    d = Sweep(2, 5)._register(tm)
    s = Sweep(0, 10, num=d)._register(tm)


    n = Nested(g, d, s)
    # n.as_list()


if 1:
    def myfunc(x, gain=1, offs=0):
        print(f'gain = {gain}')
        print(f'   x = {x}')
        print(f'offs = {offs}')
        print()
        return gain*x + offs

    x = Sweep(10, 20, step=5)
    g = Sequence([1, 10])
    tm.add_func(myfunc, x=x, gain=g, offs=0)

    tm.func.myfunc._sweep = Nested(g, x)

    #~ z = Zip(x, g)

    df = tm.func.myfunc.run()
    #     x  gain  offs  myfunc
    # 0  10     1     0      10
    # 1  15     1     0      15
    # 2  20     1     0      20
    # 3  10    10     0     100
    # 4  15    10     0     150
    # 5  20    10     0     200

if 0:
    m1 = Sweep(2, 5)._register(tm)
    m2 = Sweep(15, 25, num=3)._register(tm)



class Signal(SystemNode):
    values = InP([0,   1,   3])
    times  = InP([0, 0.1, 0.3])
    t      = InP(0)

    @state(init=0)
    def idx(idx, t, times):
        idx_next = idx + 1
        if idx_next < len(times) and t >= times[idx_next]:
            return idx_next
        else:
            return idx

    @Function
    def __return__(idx, values):
        return values[idx]


if 1:
    t = Sweep(0, 10, step=0.2)._register(tm)
    s = Signal(t=t)._register(tm)

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
    tm.func.value._overwrite = True
    tm.func.value._lazy = False

    tm.func.value._eval()
    tm.func.value._eval()
    tm.func.value._eval()

    df = tm.func.value.table
    #      start  step  idx  value
    # 0    100    11  [3]    133


if 0:
    @tm.add_func
    def new_idx():
        return [0]

    def value(start, step, idx):
        idx[0] += 1
        return start + idx[0] * step

    tm.add_func(value, start=100, step=11, idx=new_idx)
    #~ tm.func.value._overwrite = True
    tm.func.value._lazy = False

    tm.func.value._eval()
    tm.func.value._eval()
    tm.func.value._eval()

    df = tm.func.value.table
    #      start  step  idx  value
    # 0    100    11  [3]    111
    # 1    100    11  [3]    122
    # 2    100    11  [3]    133

