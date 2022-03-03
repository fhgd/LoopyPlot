import bisect
import networkx as nx
import pandas as pd

import itertools
from types import FunctionType, SimpleNamespace

import inspect
NOTHING = inspect._empty


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
    __nodes__ = []
    # todo: __dm__ = DataManager()

    def __init__(self, name='', overwrite=False):
        # todo: underscore all attributes
        self._name = name
        self._overwrite = overwrite
        self._id = len(Node.__nodes__)
        Node.__nodes__.append(self)
        self._key = f'n{self._id}'
        self._root = None
        # set by _register(tm)
        self.__tm = None
        self._iter_children = list().__iter__

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

    def _has_tm(self):
        return self.__tm is not None

    def __node__(self):
        return self

    @classmethod
    def _as_node(cls, obj):
        try:
           return obj.__node__()
        except AttributeError:
            return ValueNode(obj)

    def __call__(self):
        return self._get()

    def _needs_eval(self):
        return False

    def _get(self):
        return self._tm.dm.read(self._key)

    def _set(self, value):
        self._tm.dm.write(self._key, value, self._overwrite)
        return value

    @property
    def _last_idx(self):
        return self._tm.dm.last_idx(self._key)

    def __repr__(self):
        names = [f'n{self._id}']
        if self._name:
            names.append(self._name)
        if self._root is not None:
            names.append(repr(self._root))
        return '_'.join(names)

    def _has_results(self):
        return self in self._tm.dm


class ValueNode(Node):
    def __init__(self, value=NOTHING, name=''):
        Node.__init__(self, name)
        self._value = value

    def _register(self, tm):
        Node._register(self, tm)
        if not self._has_results() or self._get() != self._value:
            self._set(self._value)
        return self


class FuncNode(Node):
    def __init__(self, func=None, name='', overwrite=False, lazy=True, mutable=False):
        name = name if name or not func else func.__name__
        Node.__init__(self, name, overwrite)
        self._lazy = lazy
        self._sweep = Nested()
        self._mutable = mutable
        self._func = func
        self._children = []  # [(arg_node, kwarg_name), ...]
        self._iter_children = self._children.__iter__
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

    def __call__(self, nodes=None):
        if nodes is None:
            nodes = self._dep_nodes()
        for node in nodes:
            print(f'    eval {node}')
            node.__eval__()
        return self._get()

    def __eval__(self):
        # todo: this should be in a Upper-Class like DependencyNode(Node)
        args = []
        kwargs = {}
        for node, name in self._iter_children():
            value = node._get()
            if name:
                kwargs[name] = value
            else:
                args.append(value)
        retval = self._func(*args, **kwargs)
        self._set(retval)

    def _needs_eval(self):
        return not self._has_results() or not self._lazy

    def _dep_nodes(self):
        """Return depending nodes needs to be evaluated from a depth-first-search."""
        # Based on copy from dfs_labeled_edges(G, source=None, depth_limit=None)
        # in networkx/algorithms/traversal/depth_first_search.py

        # Based on http://www.ics.uci.edu/~eppstein/PADS/DFS.py
        # by D. Eppstein, July 2004.
        start = self
        depth_limit = len(start.__nodes__)
        visited = set()

        needs_eval = set()
        nodes = []

        #~ yield start, start, "forward"
        visited.add(start)
        stack = [(start, depth_limit, start._iter_children())]
        #~ print(f'START node: {start}')
        while stack:
            parent, depth_now, children = stack[-1]
            #~ print(f'{parent = }')
            try:
                child, _ = next(children)
                #~ print(f'    {child = }')
                if child._last_idx > parent._last_idx:
                    #~ print(f'        is newer than parent')
                    needs_eval.add(parent)
                if child not in visited:
                    #~ yield parent, child, "forward"
                    visited.add(child)
                    if depth_now > 1:
                        stack.append((child, depth_now - 1, child._iter_children()))
                        #~ print(f'    append child: {list(child._iter_children())}')
                #~ else:
                    #~ yield parent, child, "nontree"
                    #~ print(f'    was visited:  {child=}')
            except StopIteration:
                #~ print('    has NO further children')
                stack.pop()
                if stack:
                    #~ yield stack[-1][0], parent, "reverse"
                    grandpar = stack[-1][0]
                    if parent in needs_eval or parent._needs_eval():
                        #~ print(f'    needs EVAL, append to OUTPUT')
                        nodes.append(parent)
                        needs_eval.add(grandpar)
                        #~ print(f'    needs_eval.add: {grandpar = }')
        #~ yield start, start, "reverse"
        #~ print(f'{start = }')
        if start in needs_eval or start._needs_eval():
            #~ print(f'    needs EVAL, append to OUTPUT')
            nodes.append(start)
        return nodes

    @property
    def _mutable(self):
        return self._overwrite and not self._lazy

    @_mutable.setter
    def _mutable(self, value):
        if value:
            self._overwrite = True
            self._lazy = False

    def _add_args_kwargs(self, *args, **kwargs):
        for obj in args:
            child = self._as_node(obj), None
            self._children.append(child)
        for name, obj in kwargs.items():
            node = self._as_node(obj)
            if not node._name:
                node._name = f'{self._name}_arg_{name}'
            child = node, name
            self._children.append(child)

    def _register(self, tm):
        Node._register(self, tm)
        for node, name in self._iter_children():
            if not node._has_tm():
                node._register(tm)
            tm.g.add_edge(node, self)
        return self

    def run(self):
        self._tm.run(self)
        return self.table

    @property
    def table(self):
        names = []
        nodes = []
        for idx, (node, name) in enumerate(self._iter_children()):
            if name:
                names.append(name)
            else:
                names.append(f'arg_{idx}')
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
        self._next = FuncNode(lambda x: x, overwrite=True)
        self._next._root = self
        self._restart = ValueNode(init, 'restart')
        self._restart._root = self
        self._init = init

    def _register(self, tm):
        Node._register(self, tm)
        self._next._register(tm)
        self._restart._register(tm)
        if not self._has_results() or self._get() != self._init:
            self.reset()
        return self

    def reset(self):
        self._set(self._init)
        # reset needs to 'update' the state in dm
        # in order to trigger _new_inputs()
        #~ if self not in self._tm.dm or self._get() != init:
            #~ self._set(init)

    def next_eval(self):
        return self._next()

    def next_update(self):
        return self._set(self._next._get())

    def add_next(self, func, *args, **kwargs):
        #~ if self not in kwargs.values():
            #~ kwargs[self._name] = self
        self._next._name = 'next'
        self._next._func = func
        self._next._add_args_kwargs(*args, **kwargs)
        return self._next

    def set_restart(self, func, *args, **kwargs):
        self._restart = FuncNode(func, 'restart')
        self._restart._add_args_kwargs(*args, **kwargs)
        self._restart._root = self

    def _iter_state_updates(self):
        yield self._next


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

    def eval(self, node):
        return node()

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

    def add_func(self, func, *args, **kwargs):
        node = FuncNode(func)
        node._add_args_kwargs(*args, **kwargs)
        node._register(self)
        self.func._add(node)
        return node

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
        return obj._nodes.get(self._name, self._default).__call__()  # value


class Function:
    def __init__(self, func, name=''):
        self._name = name if name else func.__name__
        self._func = func

    def __get__(self, obj, cls=None):
        return obj._nodes.get(self._name, self._func)  # plain node


class State:
    def __init__(self, name, init, func):
        self._name = name
        self._init = init
        self._func = func

    def __get__(self, obj, cls=None):
        return obj._nodes.get(self._name, self._init)._get()  # value

    def __set__(self, obj, value):
        obj._nodes.get(self._name)._set(value)


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


class SystemNode:
    def __init__(self, *args, **kwargs):
        argitems = [repr(value) for value in args]
        argitems += [f'{name}={val!r}' for name, val in kwargs.items()]
        self._name = f'{self.__class__.__name__}({", ".join(argitems)})'
        self._root = kwargs.get('root', None)

        self._nodes = {}
        self._states = []
        self._subsys = []

        inputs = []
        self._func_nodes = []
        for cls in reversed(self.__class__.mro()):
            for name, attr in cls.__dict__.items():
                if isinstance(attr, InP):
                    inputs.append(attr)
                elif isinstance(attr, Function):
                    node = FuncNode(attr._func)
                    node._root = self
                    self._nodes[attr._name] = node
                    self._func_nodes.append(node)
                elif isinstance(attr, State):
                    name = attr._name
                    node = StateNode(name, attr._init)
                    node._root = self
                    node.add_next(attr._func)
                    self._nodes[name] = node
                    self._states.append(node)
                    self._func_nodes.append(node._next)

        # resolve inputs to __init__(*args, **kwargs)
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
            self._nodes[name] = node
            if not node._name:
                node._name = f'{self.__class__.__name__}_arg_{name}'

        self.__config__(*args, **kwargs)

        # resolve names of (keyword-) arguments with nodes
        for fnode in self._func_nodes:
            params = inspect.signature(fnode._func).parameters
            for name, param in params.items():
                if (param.kind is inspect.Parameter.VAR_POSITIONAL or
                    param.kind is inspect.Parameter.VAR_KEYWORD
                ):
                    continue
                if name not in self._nodes:
                    print(f'param {name!r} is not in nodes')
                    continue
                anode = self._nodes[name]
                fnode._add_args_kwargs(**{name: anode})

    def __config__(self, *args, **kwargs):
        pass

    def __node__(self):
        try:
            return self._nodes.get('__return__')
        except KeyError:
            return ValueNode(None, name=self.__class__.__name__)

    def _get(self):
        return self.__node__()._get()

    def __call__(self, *args, **kwargs):
        return self.__node__().__call__(*args, **kwargs)

    def __repr__(self):
        return self._name

    def set_return(self, node, *args, **kwargs):
        node = node.__node__()
        if isinstance(node, FuncNode):
            node._add_args_kwargs(*args, **kwargs)
        node._root = self
        self._nodes['__return__'] = node

    def add_subsys(self, system, name='', **kwargs):
        self._subsys.append(system)
        system._root = self
        return system

    def _register(self, tm):
        for node in self._nodes.values():
            node._register(tm)
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

    def _update(self):
        for nodes in self._next_updates():
            #~ print(f'nodes for update: {nodes}')
            for node in nodes:
                #~ print(f'update node:   {node}')
                node()
            for node in nodes:
                node._root._set(node._get())

    def update(self):
        self._update()
        return self.__node__().__call__()

    def reset(self):
        for state in self._states:
            state.reset()
        for subsys in self._subsys:
            subsys.reset()


class LoopNode(SystemNode):
    def is_valid(self):
        """Returns False if state is outside the nominal range."""
        return True

    def has_next(self):
        """Returns False on the last element."""
        return True

    def is_running(self):
        """Deprecated by has_next()."""
        return self.has_next()

    def is_interrupted(self):
        return False

    def _next(self):
        if not self.is_valid():
            #~ print('1st STOP')
            raise StopIteration
        nodes = self.__node__()._dep_nodes()
        if not nodes:
            if self.has_next() and not self.is_interrupted():
                self._update()
                nodes = self.__node__()._dep_nodes()
            else:
                #~ print('2nd STOP')
                raise StopIteration
                # Ideally this should moved into switched return-node
                #   __return__node__    :  if is_valid()
                #   raise StopIteration :  else
        return self.__node__().__call__(nodes)

    __next__ = _next

    def __iter__(self):
        if self.is_interrupted() and self.has_next():
            self._update()
        # self.reset()
        return self

    def as_list(self):
        self.reset()
        return list(self)


class CountingLoopNode(LoopNode):
    def __config__(self, *args, **kwargs):
        self._num = args[0] if args else 1

    @state(init=0)
    def idx(idx):
        return idx + 1

    def is_valid(self):
        return 0 <= self.idx < len(self)

    def has_next(self):
        return self.idx < len(self) - 1

    def __len__(self):
        return self._num

    @Function
    def __return__(idx):
        return idx


class Sweep(CountingLoopNode):
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


class Sequence(CountingLoopNode):
    items = InP()

    @Function
    def __return__(idx, items):
        return items[idx]

    def __len__(self):
        return len(self.items)


class GraphLoop(LoopNode):
    target_edge = InP()

    def __config__(self, *args):
        self.g = nx.MultiDiGraph()
        self._edges = {}  # {edge: (pre_state, post_state)}
        self._cache = {}  # {(current_node, target_node): edge}

        _node = FuncNode(self._next_edge)
        _node._root = self
        self._nodes['_next_edge'] = _node
        self._func_nodes.append(_node)

        _state = self._nodes['current_edge']
        _state.set_restart(lambda x: x, x=_state)

    def add(self, edge, pre_state='', post_state=''):
        self._edges[edge] = pre_state, post_state
        self.g.add_edge(pre_state, post_state, edge)

    def has_next(self):
        target, _ = self._edges[self.target_edge]
        _, current = self._edges[self.current_edge]
        return current != target or self.current_edge != self.target_edge

    def _calc_path(self, current, target):
        paths = nx.all_simple_edge_paths(self.g, current, target)
        for current, child, edge in min(paths, key=lambda p: len(p), default=[]):
            self._cache[current, target] = edge

    def _next_edge(self, current_edge, target_edge):
        _, current = self._edges[current_edge]
        target, _ = self._edges[target_edge]
        if current != target:
            key = current, target
            if key not in self._cache:
                self._calc_path(current, target)
            edge = self._cache[key]
        else:
            edge = target_edge
        return edge

    @state(init=0)
    def current_edge(_next_edge):
        return _next_edge

    @Function
    def __return__(current_edge):
        return current_edge


class ZipSys(LoopNode):
    def __config__(self, *args, **kwargs):
        self._subsys.extend(args)

    @Function
    def __return__(subsys):
        return subsys

    def is_running(self):
        return all(loop.is_running() for loop in self._subsys)

    def __len__(self):
        return min(len(loop) for loop in self._subsys)


class NestedSys(LoopNode):
    def __config__(self, *args):
        self._subsys.extend(args)
        self.set_return(FuncNode(self.__return__), *args)

    def __return__(self, *subsys):
        return subsys

    def idxs(self):
        return list(range(len(self._subsys)))

    def _next_updates(self):
        loops = list(self._subsys)
        idxs = self.idxs()
        n = len(idxs) - 1
        loop = loops[idxs[n]]
        while n > 0 and (not loop.has_next() or loop.is_interrupted()):
            n -= 1
            loop = loops[idxs[n]]
        #~ print(f'{n=}')
        #~ print(f'{idxs = }')
        if loop.has_next():
            next_states = set(state._next for state in loops[idxs[n]]._iter_all_states())
            #~ print(f'{next_states = }')
            yield next_states

        init_states = set()
        idxs =  self.idxs()   # refresh loops due to possible new task
        #~ print(f'{idxs = }    restarts:')
        while n < len(idxs) - 1:
            n += 1
            #~ print(f'    {n = }')
            loop = loops[idxs[n]]
            if loop.has_next() and loop.is_interrupted():
                #~ print(f'    {loop} has next and is interrupted')
                init_states.update(state._next for state in loop._iter_all_states())
            else:
                #~ print(f'    RESTART  {loop}')
                init_states.update(state._restart for state in loop._iter_all_states())
                idxs = self.idxs()   # refresh loops due to possible new task
        #~ print(f'{init_states = }')
        yield init_states

    def has_next(self):
        loops = list(self._subsys)
        return any(loops[idx].has_next() for idx in self.idxs())


class ConcatSys(NestedSys):
    def __config__(self, *loops):
        self.add_subsys(Sweep(0, len(loops) - 1))
        for loop in loops:
            self.add_subsys(loop)
        self._nodes['__return__'] = FuncNode(lambda i, v: v, name='__return__')
        self._nodes['__return__']._iter_children = self.__return__children

    def __return__children(self):
        idx = self._subsys[0]._nodes['idx']
        yield idx, ''
        yield self._subsys[idx._get() + 1].__node__(), ''

    def idxs(self):
        idx = self._subsys[0].idx
        return [0, idx + 1]

    def _register(self, tm):
        super()._register(tm)
        self._subsys[0]._register(tm)
        return self


# todo: TaskManager / TaskSequence / TaskProgram
#           Nested(tasks, task_loop, graph_loop)
class TaskProgram(NestedSys):
    def __config__(self, *tasks):
        self.add_subsys(Sweep(0, len(tasks) - 1))  # idx

        self.add_subsys(Sequence('AbC'))  # glp
        #~ self.add_subsys(LoopNode())
        #~ self.add_subsys(GraphLoop(g, on_exit, on_enter))

        for task in tasks:
            self.add_subsys(task)
        self._nodes['__return__'] = FuncNode(self.__return__)
        self._nodes['__return__']._iter_children = self.__return__children

    def __return__(self, idx, n1, n2):
        return n1, n2

    def __return__children(self):
        idx = self._subsys[0]._nodes['idx']
        yield idx, ''
        yield self._subsys[idx._get() + 2].__node__(), ''
        yield self._subsys[1].__node__(), ''

    def idxs(self):
        idx = self._subsys[0].idx
        return [0, idx + 2, 1]

    def _register(self, tm):
        super()._register(tm)
        self._subsys[0]._register(tm)
        self._subsys[1]._register(tm)
        return self


class Task(LoopNode):
    def __init__(self, func=None, name='', mainloop=None, *args, **kwargs):
        super().__init__()  # SystemNode.__init__()
        if func is not None:
            self.set_return(FuncNode(func), *args, **kwargs)
        if mainloop is None:
            mainloop = LoopNode()
            #~ mainloop = Sweep(0, 1)
        self.mainloop = self.add_subsys(mainloop)

        argitems = []
        if func:
            argitems = [func.__name__]
        #~ argitems += [f'{name}={val!r}' for name, val in kwargs.items()]
        self._name = f'{self.__class__.__name__}({", ".join(argitems)})'


    def is_running(self):
        return self.mainloop.is_running()


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
        return tuple(s.__node__()._eval() for s in self.sweeps)

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
            new_inputs += sweep.__node__()._new_inputs()
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
    class MyCounter(CountingLoopNode):
        def is_interrupted(self):
            return self.idx == 2

    c = MyCounter(10)._register(tm)
    #~ print(list(c))
    #~ print(list(c))
    #~ print(list(c))
    #~ print(list(c))

    c = MyCounter(6)._register(tm)
    x = Sweep(10, 20, step=5)._register(tm)
    n = NestedSys(x, c)._register(tm)


if 0:
    c = CountingLoopNode(3)._register(tm)
    x = Sweep(5, 15, num=3)._register(tm)
    s = Sequence('')._register(tm)

if 0:
    def my():
        return 321

    fmy = FuncNode(my)._register(tm)


if 0:
    t00 = Task()._register(tm)
    # todo: __return__ == lambda: None


if 0:
    def my():
        return 321
    t0 = Task(my)._register(tm)


if 0:
    def quad(x, gain=1, offs=0):
        return gain * x**2 + offs

    gain = Sequence([1, 2, 5])._register(tm)
    x = Sweep(0, 10, step=gain)._register(tm)

    import random
    def offs_factory():
        return random.random()
    offs = FuncNode(offs_factory, lazy=False)._register(tm)


    fn = FuncNode(quad)
    fn._add_args_kwargs(x=x, gain=gain, offs=offs)
    fn._register(tm)

if 0:
    def quad(x):
        print('eval quad')
        return x**2
    x1 = Sweep(10, 20, step=5)
    t1 = Task(quad, mainloop=x1, x=x1)._register(tm)


    def double(x):
        print('eval double')
        return 2*x
    x2 = Sequence([1, 2])
    t2 = Task(double, mainloop=x2, x=x2)._register(tm)


    def my():
        print('eval my')
        return 321
    t21 = Task(my)._register(tm)


    def countdown(x):
        print('eval countdown')
        return x
    x3 = Sequence(['3-2-1-0'])
    t3 = Task(countdown, mainloop=x3, x=x3)._register(tm)

    tp3 = TaskProgram(t1, t2, t21, t3)._register(tm)
    tp31 = TaskProgram(t21)._register(tm)
    #~ tp32 = TaskProgram()._register(tm)

    # test: tp3.as_list()
    # should be:
    #   quad, quad, quad,
    #   double, double,
    #   my,
    #   countdown

if 1:
    glp = GraphLoop(3)._register(tm)
    glp.add(0, '', '')
    glp.add(1, '',      'SLEEP')
    glp.add(2, 'SLEEP', 'ON')
    glp.add(3, 'ON',    'OFF')
    glp.add(4, 'ON',    'OFF')
    glp.add(5, 'ON',    'SLEEP')
    glp.add(6, 'OFF',   'SLEEP')
    glp.add(7, 'SLEEP', 'SLEEP')


if 0:
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


if 0:
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
    g = Sequence([3, 5])._register(tm)
    h = Sequence([127, 255])._register(tm)
    x = Sweep(10, 20, step=5)._register(tm)

    n = NestedSys(g, h, x)._register(tm)
    c = ConcatSys(x, g, h)._register(tm)
    #~ tp = TaskProgram(x, g, h)._register(tm)


if 0:
    m1 = Sweep(2, 5)._register(tm)
    m2 = Sweep(15, 25, num=3)._register(tm)

if 0:
    z = ZipSys(g, x)._register(tm)


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


if 0:
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
    # todo: idx node should have a flag for mutable datatype [x] ? [o]
    idx = ValueNode([0], name='idx')

    def value(start, step, idx):
        idx[0] += 1
        return start + idx[0] * step
    tm.add_func(value, start=100, step=11, idx=idx)
    tm.func.value._mutable = True

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


"""
todo:

Idee:

Um die Möglichkeit der config-Erstellung eines Systems zu ermöglichen,
sollte nicht eine leer Instanz (weiter-) konfiguriert werden, sondern eine neue
Klasse erzeugt werden, welche später beliebig oft instanziert werden kann!
(ala __new__???)
"""
