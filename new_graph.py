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

    def __init__(self, func=None, name=''):
        self.func = func
        self.name = name if func is None else func.__name__
        self.update = None
        self.id = Node.__count__
        Node.__count__ += 1
        self._key = ''

    @property
    def key(self):
        if self._key:
            return self._key
        #~ elif self.func:
            #~ return f'n{self.id}_{self.func.__name__}'
        #~ elif self.name:
            #~ return f'n{self.id}_{self.name}'
        else:
            return f'n{self.id}'

    def __repr__(self):
        #~ return f'n{self.id}'
        if self.func:
            return f'n{self.id}_{self.func.__name__}'
        elif self.name:
            return f'n{self.id}_{self.name}'
        else:
            return f'n{self.id}'


class ExprGraph:
    def __init__(self):
        self.g = nx.DiGraph()
        self.dm = DataManager()
        self.func = Container()

    def add_state(self, name, init):
        n1 = self._add_node(name=name)
        def func(x):
            return x
        func.__name__ = f'{name}_update'
        n2 = self.add_func(func)
        key = f'n{n1.id}_n{n2.id}_{name}'
        n1._key = key
        n2._key = key
        self.dm.write(key, init)
        n1.update = n2
        return n1

    def add_state_update(self, state, update):
        self.g.add_edge(update, state.update, arg='x')

    def add_func(self, func, **kwargs):
        func_node = self._add_node(func)
        self.func._add(func_node)
        for name, obj in kwargs.items():
            node = self._as_node(obj)
            if not node.name:
                node.name = f'{name}_{func.__name__}'
            self.g.add_edge(node, func_node, arg=name)
        return func_node

    def _as_node(self, obj):
        if hasattr(obj, 'id'):
            return obj
        else:
            node = self._add_node()
            self.dm.write(node.key, obj)
            return node

    def _add_node(self, func=None, name=''):
        node = Node(func, name)
        self.g.add_node(node)
        return node

    def eval(self, node):
        dct = nx.shortest_path_length(self.g, target=node)
        nodes = sorted(dct, key=dct.get, reverse=True)
        for n in nodes:
            if n.func is None:
                continue
            kwargs = {}
            for edge in self.g.in_edges(n):
                name = self.g.edges[edge]['arg']
                kwargs[name] = self.dm.read(edge[0].key)
            retval = n.func(**kwargs)
            self.dm.write(n.key, retval)
        return self.dm.read(node.key)

    def update_state(self, node):
        value = self.eval(node.update)
        self.dm.write(node.key, value)


class Container:
    def _add(self, node):
        name = node.name
        idx = 2
        while name in self.__dict__:
            name = f'{node.name}_{idx}'
        setattr(self, name, node)


def inc(idx):
    return idx + 1

def quad(x, gain=1, offs=0):
    return gain * x**2 + offs

def double(x):
    return 2*x

tm = ExprGraph()

idx = tm.add_state('idx', 0)
tm.add_func(inc, idx=idx)
tm.add_state_update(idx, tm.func.inc)

"""
idx.update(inc, idx=idx)


idx = tm.add_state('idx', 0)
@idx.update(idx=idx)
def inc(idx):
    return idx + 1


@tm.state('idx', init=0)
def func(idx):
    return idx + 1


@tm.state
def idx(idx=0):
    return idx + 1

"""

tm.add_func(double, x=idx)
tm.add_func(quad, x=2, gain=tm.func.double, offs=0)

print(tm.eval(tm.func.quad))
for n in range(4):
    tm.eval(tm.func.idx_update)
    print(tm.eval(tm.func.quad))

tm.dm._data
