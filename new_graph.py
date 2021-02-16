import bisect
import networkx as nx


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

    def get(self):
        return self.tm.dm.read(self.key)

    def set(self, value):
        self.tm.dm.write(self.key, value)
        return value

    def __repr__(self):
        return f'n{self.id}_{self.name}' if self.name else f'n{self.id}'


class StateNode(Node):
    def __init__(self, *args, **kwargs):
        Node.__init__(self, *args, **kwargs)
        self.next = self.tm._add_node()
        self.next.key = self.key

    def add_next(self, func, **kwargs):
        if self not in kwargs.values():
            kwargs[self.name] = self
        self.next.name = f'{self.name}_next'
        self.next.func = func
        self.tm._add_kwargs(self.next, kwargs)
        return self.next


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
        node = StateNode(self)
        node.name = name
        node.set(init)
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


class Container:
    def _add(self, node):
        name = node.name
        idx = 2
        while name in self.__dict__:
            name = f'{node.name}_{idx}'
        setattr(self, name, node)


def quad(x, gain=1, offs=0):
    return gain * x**2 + offs

def double(x):
    return 2*x

tm = TaskManager()

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

myquad.arg.x = tm.Sweep(5)
myquad.arg.x.sweep(5)

"""

tm.add_func(double, x=idx)
tm.add_func(quad, x=2, gain=tm.func.double, offs=0)

print(tm.eval(tm.func.quad))
for n in range(4):
    tm.eval(idx.next)
    print(tm.eval(tm.func.quad))

tm.dm._data
