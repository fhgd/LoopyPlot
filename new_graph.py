import bisect
import networkx as nx

import inspect
from types import FunctionType, SimpleNamespace


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

    def __init__(self, tm, func=None, name=''):
        self.tm = tm
        self.func = func
        self.name = name if func is None else func.__name__
        self.id = Node.__count__
        Node.__count__ += 1
        self.key = f'n{self.id}'

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


class StateNode(Node):
    def __init__(self, *args, **kwargs):
        self._init = kwargs.pop('init', 0)
        Node.__init__(self, *args, **kwargs)
        self.next = self.tm._add_node()
        self.next.key = self.key
        self.reset()

    def reset(self):
        self.set(self._init)

    def add_next(self, func, **kwargs):
        if self not in kwargs.values():
            kwargs[self.name] = self
        self.next.name = f'{self.name}_next'
        self.next.func = func
        self.tm._add_kwargs(self.next, kwargs)
        return self.next


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
            if n.func is None:
                continue
            kwargs = {}
            for edge in self.g.in_edges(n):
                name = self.g.edges[edge]['arg']
                kwargs[name] = edge[0].get()
            retval = n.func(**kwargs)
            n.set(retval)
        return node.get()

    def add_state(self, name, init):
        node = StateNode(self, init=init)
        node.name = name
        self.g.add_node(node)
        self.state._add(node)
        return node

    def add_func(self, func, **kwargs):
        func_node = self._add_node(func)
        self.func._add(func_node)
        self._add_kwargs(func_node, kwargs)
        return func_node

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
            node = self._add_node()
            node.set(obj)
            return node

    def _add_node(self, func=None, name=''):
        node = Node(self, func, name)
        self.g.add_node(node)
        return node

    def sweep(self, start, stop, step=1, num=None, idx=0):
        return Sweep(self, start, stop, step, num, idx)

    def mystate(self, func=None, init=0):
        def wrap(func):
            name = func.__name__
            state = self.add_state(name, init)
            state.add_next(func, **{name: state})
            return state
        if func is None:
            return wrap
        else:
            return wrap(func)


tm = TaskManager()


class Sweep:
    def __init__(self, tm, start, stop, step=1, num=None, idx=0):
        self.value = tm.add_func(self.get_value,
            idx=self.idx,
            start=start,
            stop=stop,
            step=step,
            num=num,
        )
        self.is_running = tm.add_func(self._is_running,
            idx=self.idx,
            start=start,
            stop=stop,
            step=step,
            num=num,
        )

    @tm.mystate(init=0)
    def idx(idx):
        return idx + 1

    @staticmethod
    def get_value(idx, start, stop, step, num):
        delta = stop - start
        if num is None:
            step = abs(step) if delta > 0 else -abs(step)
        else:
            div = num - 1
            div = div if div > 1 else 1
            step = float(delta) / div
        return start + step * idx

    @staticmethod
    def _is_running(idx, start, stop, step, num):
        if num is None:
            delta = stop - start
            step = abs(step) if delta > 0 else -abs(step)
            num = int(delta / step) + 1
        return 0 <= idx < num - 1

    def __call__(self):
        return self.value.eval()

    def next(self):
        return self.idx.next.eval()



def quad(x, gain=1, offs=0):
    return gain * x**2 + offs

def double(x):
    return 2*x

#~ tm = TaskManager()

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
    tm.eval(idx.next)
    print(tm.eval(tm.func.quad))

tm.dm._data


s1 = Sweep(tm, 1, 5, 0.5)
s2 = Sweep(tm, 10, 20, num=3)


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

def state(func):
    func.__is_state__ = True
    return func


class BaseSweep:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._nodes = {}
        self._tm = None

    @staticmethod
    def _identity(x):
        return x

    def register(self, tm):
        self._tm = tm
        nodes = self._nodes = {}
        func_nodes = {}
        for name, attr in self.__class__.__dict__.items():
            if name.startswith('__'):
                continue
            elif isinstance(attr, FunctionType):
                if attr.__name__.startswith('state_'):
                    state_name = attr.__name__.replace('state_', '')
                    node = tm.add_state(state_name, 0)
                    node.add_next(attr)
                    nodes[state_name] = node
                    nodes[f'{state_name}__next'] = node.next
                    func_nodes[state_name] = node
                    func_nodes[f'{state_name}__next'] = node.next
                else:
                    node = tm._add_node(attr)
                    nodes[name] = node
                    func_nodes[name] = node
            else:
                node = tm.add_func(self._identity, x=attr)
                node.name = f'arg_{name}'
                nodes[name] = node
                #~ print(f'{name}: {attr!r}')
        for node in func_nodes.values():
            if node.func is None:
                continue
            #~ print(f'check function {node.func.__name__!r}')
            params = inspect.signature(node.func).parameters
            for name, param in params.items():
                if (param.kind is inspect.Parameter.VAR_POSITIONAL or
                    param.kind is inspect.Parameter.VAR_KEYWORD):
                    continue
                if name not in nodes:
                    print(f'param {name!r} is not in nodes')
                    continue
                tm.g.add_edge(nodes[name], node, arg=name)
        return nodes


class MySweep(BaseSweep):
    #~ start: int
    #~ stop: int
    start = 1
    stop = 3
    step = 1
    num = None

    def aux(start, stop, step, num):
        if num is None:
            delta = stop - start
            _step = abs(step) if delta > 0 else -abs(step)
            _num = int(delta / _step) + 1
        else:
            div = num - 1
            div = div if div > 1 else 1
            _step = float(delta) / div
            _num = num
        return SimpleNamespace(num=_num, step=_step)

    def state_idx(idx):
        return idx + 1

    def value(idx, start, aux):
        return start + aux.step * idx

    def is_running(idx, aux):
        return 0 <= idx < aux.num - 1

    def is_finished(is_running):
        return not is_running


s = MySweep()
s.register(tm)



def identity(x):
    return x


def parse_defaults(func):
    defaults = {}
    for name, param in inspect.signature(func).parameters.items():
        if (param.kind is not inspect.Parameter.VAR_POSITIONAL and
            param.kind is not inspect.Parameter.VAR_KEYWORD):
            if param.default is param.empty:
                defaults[name] = None
            else:
                defaults[name] = param.default
    return defaults


def register(tm, cls, **kwargs):
    nodes = {}
    for name, attr in cls.__dict__.items():
        if isinstance(attr, FunctionType):
            #~ if getattr(attr, '__is_state__', False):
            if attr.__name__.startswith('state_'):
                state_name = attr.__name__.replace('state_', '')
                node = tm.add_state(state_name, 0)
                node.add_next(attr)
                nodes[state_name] = node
                nodes[f'{state_name}__next'] = node.next
            else:
                node = tm._add_node(attr)
                nodes[name] = node
    for name, node in dict(**nodes).items():
        if node.func is None:
            continue
        for arg, default in parse_defaults(node.func).items():
            #~ print(name, arg)
            if arg not in nodes:
                arg_value = kwargs.get(arg, default)
                arg_node = tm.add_func(identity, x=arg_value)
                #~ print(f'{arg}: {arg_value!r}')
                arg_node.name = f'arg_{arg}'
                nodes[arg] = arg_node
            arg_node = nodes[arg]
            tm.g.add_edge(arg_node, node, arg=arg)
    return nodes
