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


def quad(x, gain=1, offs=0):
    return gain * x**2 + offs

def double(x):
    return 2*x

class TestManager:
    def __init__(self, graph={}):
        self._graph = graph

    def evaluate(self, func, *args):
        kwargs = {}
        for name, (fn, *fargs) in self._graph[func].items():
            print(f'{name}: {fn.__name__}({", ".join(fargs)})')
            try:
                kwargs[name] = self.evaluate(fn, *fargs)
            except KeyError:
                kwargs[name] = fn(*fargs)
        retval = func(*args, **kwargs)
        # todo:  dm.write(f'{func.__name__}.__return__', retval)
        return retval


dm = DataManager()
dm.write('quad.x', 2)
dm.write('quad.offs', 0)
dm.write('double.x', 0.5)

tm = TestManager(graph={
    quad: {
        'x':    [dm.read, 'quad.x'],
        'offs': [dm.read, 'quad.offs'],
        'gain': [double],
    },
    double: {
        'x': [dm.read, 'double.x'],
    }
})

value = tm.evaluate(quad)
dm.write('quad._return', value)
print(f'quad = {value}')


