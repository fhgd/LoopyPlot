import os
import sys
import inspect, ast
from textwrap import dedent
from time import sleep
from sortedcontainers import SortedDict, SortedSet
from collections import OrderedDict, namedtuple
from functools import wraps
import json

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import colorConverter
from colorsys import rgb_to_hls, hls_to_rgb

import itertools
try:
    from itertools import zip_longest
except AttributeError:
    from itertools import izip_longest as zip_longest

from . import utils
log = utils.get_plain_logger(__name__)
utils.enable_logger(log, 'info', 'short')

from . import plotmanager


class BaseSweepIterator:
    """Base class for sweeped iterators.

    >>> class MySweep(BaseSweepIterator):
    ...     def __len__(self):
    ...         return 5
    ...
    ...     def get_value(self, idx):
    ...         return idx
    ...
    >>> s = MySweep()
    >>> list(s)
    [0, 1, 2, 3, 4]
    >>> list(s)
    []
    >>> s.idx
    4
    >>> s.idx = 4
    >>> list(s)
    [4]

    >>> s[2]
    2
    >>> s[-2]
    3
    >>> s[1:3]
    [1, 2]
    >>> s[::2]
    [0, 2, 4]
    """
    def __init__(self):
        self._idx = 0
        self._is_initialized = True

    def __len__(self):
        return NotImplemented

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, value):
        self._validate_idx(value)
        self._idx = value
        self._is_initialized = True

    def next_idx(self):
        self._idx += 1
        return self.idx

    def get_value(self, idx):
        return NotImplemented

    def __getitem__(self, obj):
        if isinstance(obj, slice):
            values = []
            start, stop, step = obj.indices(len(self))
            self._validate_idx(start)
            self._validate_idx(stop)
            idxs = range(start, stop, step)
            return [self.get_value(idx) for idx in idxs]
        elif isinstance(obj, int):
            idx = len(self) + obj if obj < 0 else obj
            self._validate_idx(idx)
            return self.get_value(idx)
        msg = 'index must be either int or slice'
        raise ValueError(msg)

    @property
    def value(self):
        return self.get_value(self.idx)

    def is_running(self):
        return 0 <= self.idx < len(self) - 1

    def is_finished(self):
        return not self.is_running()

    def next_value(self):
        if self._is_initialized:
            self._is_initialized = False
        else:
            if self.is_running():
                self.next_idx()
            else:
                raise StopIteration
        return self.value

    @property
    def idx_next(self):
        # used for progress logging
        if self._is_initialized:
            return self.idx
        else:
            return self.idx + 1
        # ToDo: remove self._is_initialized
        #
        # def next_value(self):
        #     idx = idx_next
        #     idx_next = self.get_next_idx(idx)
        #     return self.get_value(idx)

    def __next__(self):
        return self.next_value()

    def __iter__(self):
        return self

    def as_list(self):
        self.reset()
        return list(self)

    def reset(self):
        self.idx = 0
        self._is_initialized = True

    def _validate_idx(self, value):
        if not isinstance(value, int):
            raise ValueError('idx must be an integer')
        elif value < 0:
            raise ValueError('idx must be positive')
        elif value > len(self):
            msg = 'idx must be less than {}'.format(len(self))
            raise ValueError(msg)


class Sweep(BaseSweepIterator):
    """Sweep linearly from start to stop.

    >>> Sweep(1, 3).as_list()
    [1, 2, 3]
    >>> Sweep(1, 3, step=0.5).as_list()
    [1.0, 1.5, 2.0, 2.5, 3.0]
    >>> Sweep(1, 3, num=5)
    Sweep(1, 3, step=0.5)
    >>> Sweep(1, 3, num=5).as_list()
    [1.0, 1.5, 2.0, 2.5, 3.0]
    """
    def __init__(self, start, stop, step=1, num=None, idx=0):
        self.start = start
        self.stop = stop

        delta = self.stop - self.start
        if num is None:
            self.step = abs(step) if delta > 0 else -abs(step)
            num = round(delta / self.step)
            self.num = num + 1
        else:
            self.num = num
            div = self.num - 1
            div = div if div > 1 else 1
            self.step = float(delta) / div

        super().__init__()
        self.idx = idx

    def __len__(self):
        return self.num

    def get_value(self, idx):
        return self.start + self.step * idx

    @property
    def min(self):
        return min(self.start, self.stop)

    @property
    def max(self):
        return max(self.start, self.stop)

    def __repr__(self):
        args = []
        args.append(repr(self.start))
        args.append(repr(self.stop))
        for name in ['step']:
            arg = '{}={}'.format(name, repr(getattr(self, name)))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


class Iterate(BaseSweepIterator):
    """Iterate over given sequence of elements.

    >>> s = Iterate(5, 2, 4, 7)
    >>> s
    Iterate(5, 2, 4, 7)
    >>> len(s)
    4
    >>> s.value
    5
    >>> s.idx
    0
    >>> s.as_list()
    [5, 2, 4, 7]
    >>> s.value
    7
    >>> s.idx
    3
    """
    def __init__(self, *items):
        self.items = list(items)
        super().__init__()

    def __len__(self):
        return len(self.items)

    def get_value(self, idx):
        return self.items[idx]

    @property
    def min(self):
        try:
            return min(self.items)
        except Exeption:
            return None

    @property
    def max(self):
        try:
            return max(self.items)
        except Exeption:
            return None

    def __repr__(self):
        args = []
        for item in self.items:
            arg = '{}'.format(repr(item))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


class Nested(BaseSweepIterator):
    """Iterate over nested sweeps.

    >>> s1 = Sweep(1, 2)
    >>> s2 = Iterate(10, 20)
    >>> n = Nested(s1, s2)
    >>> n.as_list()
    [(1, 10), (2, 10), (1, 20), (2, 20)]
    """
    def __init__(self, *sweeps):
        self.sweeps = list(sweeps)
        super().__init__()

    def __len__(self):
        value = 1
        for sweep in self.sweeps:
            value *= len(sweep)
        return value

    @property
    def idx(self):
        sweeps = self.sweeps
        if not sweeps:
            return 0
        idx = 0
        num = 1
        for sweep in sweeps:
            idx += sweep.idx * num
            num *= len(sweep)
        return idx

    @idx.setter
    def idx(self, value):
        for sweep, sidx in self._get_sweep_idx(value).items():
            sweep.idx = sidx
        self._is_initialized = True

    def get_value(self, idx):
        dct = self._get_sweep_idx(idx)
        values = []
        for sweep in self.sweeps:
            sidx = dct[sweep]
            values.append(sweep[sidx])
        return tuple(values)

    def _get_sweep_idx(self, idx):
        self._validate_idx(idx)
        num = len(self)
        sweeps = {}
        for sweep in reversed(self.sweeps):
            num = num // len(sweep)
            sweeps[sweep] = idx // num
            idx = idx % num
        return sweeps

    def reset(self):
        for sweep in self.sweeps:
            sweep.reset()
        self._is_initialized = True

    def next_idx(self):
        sweeps = self.sweeps
        n = 0
        while sweeps[n].is_finished():
            n += 1
        sweeps[n].next_idx()
        n -= len(sweeps)
        sweeps = self.sweeps    # refresh sweep list
                                # due to possible new task
        while n > -len(sweeps):
            n -= 1
            sweep = sweeps[n]
            if sweep.is_finished():
                sweep.reset()
                sweeps = self.sweeps    # refresh sweep list
            elif sweep._is_initialized:
                sweep.next_idx()
                sweeps = self.sweeps    # refresh sweep list
        return self.idx

    def __repr__(self):
        args = [', '.join('{!r}'.format(sweep) for sweep in self.sweeps)]
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


class Zip(BaseSweepIterator):
    """Zip several sweep together

    >>> s1 = Sweep(1, 3)
    >>> s2 = Iterate(10, 20, 30)
    >>> Zip(s1, s2).as_list()
    [(1, 10), (2, 20), (3, 30)]
    """
    def __init__(self, *sweeps):
        self.sweeps = list(sweeps)
        super().__init__()

    def __len__(self):
        return min(len(sweep) for sweep in self.sweeps)

    @BaseSweepIterator.idx.setter
    def idx(self, value):
        BaseSweepIterator.idx.fset(self, value)
        for sweep in self.sweeps:
            sweep.idx = value

    def next_idx(self):
        self._idx += 1
        for sweep in self.sweeps:
            sweep.idx = self._idx
        return self._idx

    def get_value(self, idx):
        return tuple(sweep[idx] for sweep in self.sweeps)

    @property
    def min(self):
        return tuple(sweep.min for sweep in self.sweeps)

    @property
    def max(self):
        return tuple(sweep.max for sweep in self.sweeps)

    def reset(self):
        super().reset()
        for sweep in self.sweeps:
            sweep.reset()

    def __repr__(self):
        args = []
        for sweep in self.sweeps:
            arg = '{!r}'.format(sweep)
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


class Concat(BaseSweepIterator):
    """Concatenate multiple sweeps.

    >>> c = Concat(Sweep(1, 3), Iterate(10, 20))
    >>> list(c)
    [1, 2, 3, 10, 20]
    >>> c.idx
    4
    >>> c.idx = 2
    >>> list(c)
    [3, 10, 20]
    >>> len(c)
    5

    >>> c.idx = 0
    >>> [s.idx for s in c.sweeps.items]
    [0, 0]
    """
    def __init__(self, *sweeps):
        self.sweeps = Iterate(*sweeps)
        super().__init__()

    def __len__(self):
        return sum(len(sweep) for sweep in self.sweeps.items)

    def get_value(self, idx):
        slen = 0
        for sweep in self.sweeps.items:
            slen += len(sweep)
            if slen > idx:
                break
        sidx = idx - slen + len(sweep)
        return sweep[sidx]

    @property
    def idx(self):
        sweeps = self.sweeps.items[:self.sweeps.idx + 1]
        return sum(sweep.idx for sweep in sweeps) + self.sweeps.idx

    @idx.setter
    def idx(self, idx):
        self._validate_idx(idx)
        slen = 0
        for num, sweep in enumerate(self.sweeps.items):
            slen += len(sweep)
            if slen > idx:
                break
            sweep.idx = len(sweep) - 1
        sweep.idx = idx - slen + len(sweep)
        self.sweeps.idx = num
        for sweep in self.sweeps.items[num + 1:]:
            sweep.idx = 0
        self._is_initialized = True

    def next_idx(self):
        sweep = self.sweeps.value
        if sweep.is_running():
            sweep.next_idx()
        else:
            self.sweeps.next_idx()
        return self.idx

    def reset(self):
        for sweep in self.sweeps.items:
            sweep.reset()
        self.sweeps.reset()
        self._is_initialized = True

    def append(self, sweep):
        self.sweeps.items.append(sweep)
        self.reset()

    def __repr__(self):
        args = []
        for sweep in self.sweeps.items:
            arg = '{!r}'.format(sweep)
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


class ConcatHold(Concat):
    def next_idx(self):
        self.sweeps.value.next_idx()
        return self.idx

    def next_sweep(self):
        self.sweeps.next_idx()
        self.sweeps.value.reset()

    def is_running(self):
        return self.sweeps.value.is_running()

    def reset(self):
        self.sweeps.value.reset()
        self._is_initialized = True

    def append(self, sweep):
        self.sweeps.items.append(sweep)


class ConcatPointer(BaseSweepIterator):
    """Concatenate multiple sweeps from pointers.

    >>> p1 = SweepPointer(Sweep(1, 3))
    >>> p2 = SweepPointer(Iterate(10, 20, 50))
    >>> c = ConcatPointer(p1, p2)
    >>> c[1]
    2
    >>> c[4]
    20

    >>> list(c)
    [1, 2, 3, 10, 20, 50]
    >>> c.idx
    5
    >>> c.idx = 2
    >>> list(c)
    [3, 10, 20, 50]
    >>> len(c)
    6

    >>> c.idx = 0
    >>> [ptr.sweep.idx for ptr in c._pointers]
    [0, 0]

    ToDo:
        * Generalize Concat class in order to inherit ConcatPointer
          from Concat
    """
    def __init__(self, *pointers):
        self._pointers = list(pointers)
        self._iptr = 0
        super().__init__()

    def append(self, pointer):
        self._pointers.append(pointer)

    def __len__(self):
        return sum(len(ptr.sweep) for ptr in self._pointers)

    def get_value(self, idx):
        slen = 0
        for ptr in self._pointers:
            slen += len(ptr.sweep)
            if slen > idx:
                break
        sidx = idx - slen + len(ptr.sweep)
        return ptr.sweep[sidx]

    @property
    def value(self):
        return self._pointers[self._iptr].sweep.value

    @property
    def idx(self):
        value = sum(len(ptr.sweep) for ptr in self._pointers[:self._iptr])
        return value + self._pointers[self._iptr].sweep.idx

    @idx.setter
    def idx(self, idx):
        self._validate_idx(idx)
        slen = 0
        for num, ptr in enumerate(self._pointers):
            sweep = ptr.sweep
            slen += len(sweep)
            if slen > idx:
                break
            sweep.idx = len(sweep) - 1
        sweep.idx = idx - slen + len(sweep)
        self._iptr = num
        for ptr in self._pointers[num + 1:]:
            ptr.sweep.idx = 0
        self._is_initialized = True

    def next_idx(self):
        sweep = self._pointers[self._iptr].sweep
        if sweep.is_running():
            sweep.next_idx()
        else:
            self._iptr += 1
            #~ self._pointers[self._iptr].sweep.reset()
        return self.idx

    def reset(self):
        for ptr in self._pointers:
            if hasattr(ptr, 'sweep') or 1:
                ptr.sweep.reset()
        self._iptr = 0
        self._is_initialized = True

    def __repr__(self):
        args = []
        for ptr in self._pointers:
            arg = '{!r}'.format(ptr.sweep)
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


def _value_str(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, int):
        return '{}'.format(value)
    elif isinstance(value, (float, complex)):
        return '{:.7}'.format(value)
    elif isinstance(value, list):
        return '[{:.4g}, ...]'.format(value[0])
    elif isinstance(value, tuple):
        return '({:.4g}, ...)'.format(value[0])
    elif isinstance(value, np.ndarray):
        with NumpyPrettyPrint(precision=2, threshold=5, edgeitems=1):
            value_str = repr(value)
        return value_str
    else:
        return repr(value)


class ReturnValue:
    def __init__(self, name, unit=None, task=None):
        self.name = name
        self._unit = unit
        self._task = task
        self._cache = []

    def _reset(self):
        self._cache = []

    def _to_dict(self):
        dct = OrderedDict()
        dct['__param__'] = self.__class__.__name__
        dct['name'] = self.name
        dct['task'] = self._task.name
        return dct

    def __repr__(self):
        return '<{}.returns.{}>'.format(self._task.name, self.name)

    @property
    def _uname(self):
        unit = ' / {}'.format(self._unit) if self._unit else ''
        return self.name + unit

    def get_cache(self, cidxs=None):
        if cidxs is None:
            return self._cache
        try:
            return [self.get_cache(cidx) for cidx in cidxs]
        except TypeError:
            return self._cache[cidxs]

    # ReturnValue should behave like a BaseSweepIterator because
    # it can be used for an (input) argument of an other task
    #
    # ToDo: Instead, try to use tasksweep.ResultSweep
    #       in order to obtain clean interfaces!

    def __getitem__(self, idx):
        return self._cache[idx]

    @property
    def idx(self):
        return self._task.nested.idx

    def __len__(self):
        return len(self._task.args._nested)

    @property
    def value(self):
        try:
            return self._cache[-1]
        except IndexError:
            return None

    @value.setter
    def value(self, value):
        self._cache.append(value)

    def as_table(self, idxs=None, include=[], hide_const=False,
                 as_df=True):
        if isinstance(idxs, int):
            idxs = [idxs]
        params = OrderedDict()
        for name, arg in self._task.args:
            arg_keys = arg._states.keys()
            if not hide_const or len(arg_keys) > 1 or len(self._cache) < 2:
                params[name] = arg.get_cache(idxs)

        params[self.name] = self.get_cache(idxs)

        for param in include:
            if isinstance(param, (list, tuple)):
                param, via = param[0], param[1:]
            else:
                via = []
            path = self._task.get_path(param, via)
            values = self._task.get_value_from_path(path, idxs)
            name = '{}.{}.{}'.format(
                param._task.name,
                'args' if isinstance(param, Argument) else 'results',
                param.name
            )
            params[name] = values
        if idxs:
            params['idxs'] = list(idxs)
        if not as_df:
            return params
        df = pd.DataFrame(params)
        if idxs:
            df = df.set_index('idxs')
            df.index.name = None
        return df

    def select_by_idx(self, **kwargs):
        task = self._task
        names = OrderedDict()
        for name, value in kwargs.items():
            if name in task.args:
                arg = task.args[name]
                idx = value
                names[name] = arg._states[idx]
        idxs = set.intersection(*names.values())
        return sorted(idxs)

    def select(self, **kwargs):
        task = self._parent
        names = OrderedDict()
        for name, value in kwargs.items():
            if name in task.args:
                arg = task.args[name]
                values = np.asarray(arg._results)
                cidx = np.abs(values - value).argmin()
                idx = arg._cache[cidx]
                names[name] = idx
        idxs = self.select_by_idx(**names)
        return self.as_table(idxs)


class ConstantParam:
    __slots__ = ['_value']
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @property
    def idx(self):
        return 0

    @property
    def state(self):
        return (None, 0)

    def get_value(self, state):
        return self._value

    def __len__(self):
        return 1

    def __repr__(self):
        args = ['{!r}'.format(self._value)]
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


class DependParam:
    def __init__(self, param, squeeze=[]):
        self.param = param
        self.squeeze = squeeze
        super().__init__()
        self.tstate = []


    def __len__(self):
        return len(self.tstate)
        #~ return len(self._task.nested)

    @property
    def idx(self):
        return len(self) - 1
        #~ return self._task.nested.idx

    @property
    def state(self):
        task = self.param._task
        return (task.clen, 0)

    @property
    def value(self):
        task = self.param._task
        if task.clen not in self.tstate:
            self.tstate.append(task.clen)
        idxs, _ = task.args.select_by_sweep(squeeze=self.squeeze)
        return self.param.get_cache(idxs[0])

    def __getitem__(self, item):
        #~ return self.value
        idx = item
        cidx = self.tstate[idx]
        return self.get_value((cidx, 0))

    def get_value(self, state):
        task = self.param._task
        cidx = state[0] - 1
        if cidx < 0:
            msg = 'task {} has no results, try .run()'.format(task)
            raise ValueError(msg)
        idxs,_ = task.args.select_by_sweep(squeeze=self.squeeze, cidx=cidx)
        return self.param.get_cache(idxs[0])


class ConstantPointer:
    __slots__ = ['_value']

    def __init__(self, value):
        self._value =  value

    @property
    def value(self):
        return self._value

    @property
    def state(self):
        return None

    @state.setter
    def state(self, value):
        if value is not None:
            msg = 'state of {} must be: None'
            msg = msg.format(self.__class__.__name__)
            raise ValueError(msg)

    def get_value(self, state):
        return self._value


class SweepPointer:
    def __init__(self, sweep):
        self._sweep = sweep

    @property
    def value(self):
        return self._sweep.value

    @property
    def state(self):
        return self._sweep.idx

    @state.setter
    def state(self, value):
        self._sweep.idx = value

    def get_value(self, state):
        return self._sweep[state]

    @property
    def sweep(self):
        return self._sweep


class SweepFactoryPointer:
    """ Like Sweep but accepts dependencies.

    >>> @Task
    ... def binary(x):
    ...     y = 2**x
    ...     return y
    >>> binary.args.x.sweep(1, 3)
    >>> binary.run(0)

    >>> @Task
    ... def factory(start, stop, step, num):
    ...     sweep = Sweep(start, stop, step, num)
    ...     return sweep
    >>> factory.args.start.value = 1
    >>> factory.args.stop.value = 4
    >>> factory.args.num.depends_on(binary.returns.y)

    >>> def func(a):
    ...     return
    >>> task = Task(func)
    >>> ptr = SweepFactoryPointer(factory)
    >>> task.args.a._ptr = ptr
    >>> task.args._add_sweeped_arg(task.args.a)
    >>> task.args
    a = 1.0 (1/2):  Sweep(1, 4, step=3.0)
    >>> task.run()
    >>> task.args.as_table()
         a
    0  1.0
    1  4.0
    >>> binary.run(0)
    >>> task.args._configure()
    >>> task.args
    a = 1.0 (1/4):  Sweep(1, 4, step=1.0)
    >>> task.run()
    >>> task.args.as_table()
         a
    0  1.0
    1  4.0
    2  1.0
    3  2.0
    4  3.0
    5  4.0
    """

    def __init__(self, factory):
        self._factory = factory
        self._tstate = []
        self._sweeps = {}
        self.configure()

    def configure(self):
        self._factory.args._configure()
        if self._factory.args._has_changed():
            self._factory.call()
            sweep = self._factory.returns.sweep.value
            tstate = self._factory.clen
            self._sweeps[tstate] = sweep
            self._tstate.append(tstate)

    @property
    def state(self):
        return len(self._tstate) - 1, self.sweep.idx

    def get_value(self, state):
        tidx, sidx = state
        tstate = self._tstate[tidx]
        sweep = self._sweeps[tstate]
        return sweep[sidx]

    @property
    def value(self):
        return self.get_value(self.state)

    @property
    def sweep(self):
        tstate = self._tstate[-1]
        return self._sweeps[tstate]

    @property
    def task(self):
        return self._factory

    # ToDo: merge into self.get_value
    def get_task_cidx(self, state):
        tidx, sidx = state
        tstate = self._tstate[tidx]
        cidx = tstate - 1
        return cidx


def config(func):
    @wraps(func)
    def myfunc(self, *args, **kwargs):
        try:
            task = self._task
        except AttributeError:
            task = self
        config = func, self, args, kwargs
        task._config.append(config)
        func(self, *args, **kwargs)
    return myfunc


class Argument:
    def __init__(self, name, default=None, unit=None, task=None):
        self.name = name
        self._default = ConstantPointer(default)
        self._unit = unit
        self._ptrs = ConcatPointer(self._default)
        self._cache = []
        self._states = {}
        self._task = task

    def _to_dict(self):
        dct = OrderedDict()
        dct['__param__'] = self.__class__.__name__
        dct['name'] = self.name
        dct['task'] = self._task.name
        return dct

    def _reset(self):
        self._cache = []
        self._states = {}

    def __repr__(self):
        return '<{}.args.{}>'.format(self._task.name, self.name)

    @property
    def _uname(self):
        unit = ' / {}'.format(self._unit) if self._unit else ''
        return self.name + unit

    @property
    def _ptr(self):
        return self._ptrs._pointers[self._ptrs._iptr]

    @_ptr.setter
    def _ptr(self, value):
        self._ptrs._pointers = [value]

    def get_arg_state(self, cidx):
        return self._cache[cidx]

    @property
    def state(self):
        iptr = self._ptrs._iptr
        ptr = self._ptrs._pointers[iptr]
        return iptr, ptr.state

    @state.setter
    def state(self, value):
        iptr, state = value
        ptr = self._ptrs._pointers[iptr]
        log.debug('{!r}: state={}: for ptr={}'.format(self, value, ptr))
        ptr.state = state

    def get_value(self, arg_state):
        iptr, state = arg_state
        return self._ptrs._pointers[iptr].get_value(state)

    def get_cache(self, cidxs=None):
        if cidxs is None:
            cidxs = range(len(self._cache))
        try:
            return [self.get_cache(cidx) for cidx in cidxs]
        except TypeError:
            arg_state = self.get_arg_state(cidxs)
            return self.get_value(arg_state)

    def get_depend_cidx(self, cidxs):
        try:
            return [self.get_depend_cidx(cidx) for cidx in cidxs]
        except TypeError:
            iptr, state = self.get_arg_state(cidxs)
            return self._ptrs._pointers[iptr].get_task_cidx(state)

    @property
    def _tasks(self):
        tasks = []
        for ptr in self._ptrs._pointers:
            if hasattr(ptr, 'task'):
                tasks.append(ptr.task)
        return tasks

    def _get_squeezed_args(self):
        args = []
        for ptr in self._ptrs._pointers:
            if hasattr(ptr, '_param'):
                task = ptr._param._task
                for name in ptr._squeeze:
                    arg = task.args[name]
                    if arg not in args:
                        args.append(arg)
        return args

    @config
    def use_default(self):
        self._ptr = self._default

    @property
    def value(self):
        return self._ptr.value

    @value.setter
    def value(self, value):
        self._set_value(value)

    @config
    def _set_value(self, value):
        self._ptr = ConstantPointer(value)
        self._cache.clear()
        self._task.args._remove_sweeped_arg(self)

    def _capture_idx(self):
        state = self.state
        self._cache.append(state)
        iset = self._states.setdefault(state, set())
        iset.add(len(self._cache) - 1)

    # ToDo: this could be in BasePointer
    def configure(self):
        for ptr in self._ptrs._pointers:
            if hasattr(ptr, 'configure'):
                ptr.configure()

    @config
    def sweep(self, start, stop, step=1, num=None, zip='', concat=False):
        """Sweep linearly from start to stop incremented by step.

        >>> @Task
        ... def binary(x):
        ...     y = 2**x + 1
        ...     return y
        >>> binary.args.x.sweep(1, 2)
        >>> binary.run(0)

        >>> @Task
        ... def task(a):
        ...     return
        >>> task.args.a.sweep(1, 3, num=binary.returns.y)
        >>> task.run()
        >>> binary.run()
        >>> binary.returns.as_table()
           x   y
        0  1   3
        1  2   5
        >>> task.args._configure()
        >>> task.run()
        >>> task.args.as_table()
             a
        0  1.0
        1  2.0
        2  3.0
        3  1.0
        4  1.5
        5  2.0
        6  2.5
        7  3.0
        """
        @Task
        def factory(start, stop, step, num):
            sweep = Sweep(start, stop, step, num)
            return sweep
        kwargs = dict(start=start,
                      stop=stop,
                      step=step,
                      num=num)
        use_factory = False
        for name, value in kwargs.items():
            if isinstance(value, (Argument, ReturnValue)):
                factory.args[name].depends_on(value)
                use_factory = True
            else:
                factory.args[name].value = value

        if use_factory:
            ptr = SweepFactoryPointer(factory)
        else:
            ptr = SweepPointer(Sweep(start, stop, step, num))
        if concat:
            self._ptrs.append(ptr)
        else:
            self._ptr = ptr

        self._task.args._add_sweeped_arg(self)
        if zip:
            self._task.args.zip(self, zip)

    @config
    def iterate(self, *items, zip='', concat=False):
        sweep = Iterate(*items)
        if concat:
            self._ptrs.append(SweepPointer(sweep))
        else:
            self._ptr = SweepPointer(sweep)
        self._task.args._add_sweeped_arg(self)
        if zip:
            self._task.args.zip(self, zip)

    @config
    def zip(self, *items):
        """Zip other argument item sweeps to argument sweep.

        >>> @Task
        ... def task(a, b, c, d):
        ...     return
        >>> task.args.a.iterate(1, 2)
        >>> task.args.b.iterate(11, 22)
        >>> task.args.c.iterate(100, 200)
        >>> task.args.d.iterate(111, 222)

        >>> task.args.b.zip('d')
        >>> task.args.a.iterate(1, 2)

        >>> task.run()
        >>> task.args.as_table()
           a   b    c    d
        0  1  11  100  111
        1  1  22  100  222
        2  1  11  200  111
        3  1  22  200  222
        4  2  11  100  111
        5  2  22  100  222
        6  2  11  200  111
        7  2  22  200  222
        """
        self._task.args.zip(self, *items)

    @config
    def depends_on(self, param, concat=False):
        #~ self._task.args._remove_sweeped_arg(self)
        task = param._task
        if self._task.args._last_tasksweep is None:
            msg = '{} is not a depending task'.format(task)
            log.error(msg)
            msg = "try: {}.args.add_depending_task({}, squeeze='')"
            msg = msg.format(self._task.name, task.name)
            log.info(msg)
            return
        elif task is not self._task.args._last_tasksweep.task:
            msg = '{} was not the last added depending task'.format(task)
            log.error(msg)
            msg = "try: {}.args.add_depending_task({}, squeeze='')"
            msg = msg.format(self._task.name, task.name)
            log.info(msg)
            return
        ptr = DependParamPointer(param, self._task.args._last_tasksweep)
        if concat:
            self._ptrs.append(ptr)
        else:
            self._ptr = ptr
        self._task.args._tasksweep_args[self] = ptr._tasksweep
        ptr._tasksweep.args.append(self)
        self._task.args._add_sweeped_arg(self)
        args = self._task.args._last_tasksweep_args
        args.append(self)
        if len(args) > 1:
            self._task.args.zip(*self._task.args._last_tasksweep_args)


class TaskSweep(BaseSweepIterator):
    def __init__(self, task, squeeze=''):
        self.task = task
        self.squeeze = squeeze if squeeze else []
        self.args = []
        self.clen = 0
        self.last_clen = 0
        super().__init__()

        # used for caching
        self._key_paths = []
        self._cidxs = []    # loop over cidxs of task
        self._states = {}   # (last_clen, clen): [{cidxs_of_this_key}, ..]
                            #                     cidx_of_key

    def __repr__(self):
        args = []
        args.append(repr(self.task))
        args.append('squeeze={!r}'.format(self.squeeze))
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def get_value(self, idx):
        cidx = self._cidxs[idx]
        return self.last_clen, self.clen, cidx

    def __len__(self):
        return len(self._cidxs)

    def configure(self):
        if self.clen != self.task.clen:
            self.last_clen = self.clen
            self.clen = self.task.clen
            self._key_paths = self.get_key_paths()
            self._cidxs = self.create_states(self.last_clen, self.clen)
            self.idx = 0

    def create_states(self, last_clen, clen):
        states = []
        cidxs = []
        keys = {}
        for cidx in range(last_clen, clen):
            key = self.get_key(cidx)
            if key not in keys.keys():
                cidxs.append(cidx)
            keys.setdefault(key, set()).add(cidx)
            states.append(keys[key])
        self._states[self.last_clen, self.clen] = states
        return cidxs

    def get_key_paths(self, squeezed_paths=[]):
        sq_paths = self.squeeze + squeezed_paths
        paths = []
        _tasksweeps = set()
        for name, arg in self.task.args:
            arg_path = [arg]
            if arg in self.task.args._tasksweep_args.keys():
                # arg is depending argument
                tasksweep = self.task.args._tasksweep_args[arg]
                if arg_path in sq_paths or tasksweep in _tasksweeps:
                    continue
                _tasksweeps.add(tasksweep)
                for path in tasksweep.get_key_paths(sq_paths):
                    full_path = [arg] + path
                    if full_path not in sq_paths:
                        paths.append(full_path)
            elif arg_path not in sq_paths:
                # arg is local argument
                paths.append(arg_path)
        return paths

    def _get_cidx_from_path(self, path, cidx):
        try:
            return [self._get_cidx_from_path(path, idx) for idx in cidx]
        except TypeError:
            task = path[0]._task
            for arg in path[:-1]:
                if task == arg._task:
                    task = arg._ptr.task
                else:
                    msg = 'task {} != {}'.format(task, arg._task)
                    raise ValueError(msg)
                cidx = arg.get_depend_cidx(cidx)
        return cidx

    def get_key(self, cidx):
        if not self._key_paths:
            self._key_paths = self.get_key_paths()
        states = []
        for path in self._key_paths:
            _cidx = self._get_cidx_from_path(path, cidx)
            try:
                state = path[-1].get_arg_state(_cidx[0])
            except TypeError:
                state = path[-1].get_arg_state(_cidx)
            states.append(state)
        return tuple(states)

    def get_cidxs(self, last_clen, clen, cidx):
        if (last_clen, clen) not in self._states:
            msg = '{!r}: create states for (last_clen, clen) = ({}, {})'
            msg = msg.format(self, last_clen, clen)
            log.warning(msg)
            self.create_states(last_clen, clen)
        return sorted(self._states[last_clen, clen][cidx - last_clen])


class DependParamPointer:
    def __init__(self, param, tasksweep):
        msg = 'param and tasksweep must have the same task'
        assert param._task is tasksweep.task, msg
        self._param = param
        self._tasksweep = tasksweep

    @property
    def state(self):
        return self._tasksweep.value

    def get_value(self, state):
        try:
            cidxs = self._tasksweep.get_cidxs(*state)
            if len(cidxs) > 1:
                return np.array(self._param.get_cache(cidxs))
            else:
                return self._param.get_cache(cidxs[0])
        except IndexError:
            msg = 'warning: no results from task {}, try {}.run()'
            task = self._param._task
            msg = msg.format(task, task.name)
            log.warning(msg)
            return np.nan

    @property
    def value(self):
        return self.get_value(self.state)

    def _get_key_dict(self, cidx):
        return self._param._task.args._get_key_dict(cidx)

    @property
    def sweep(self):
        return self._tasksweep

    def configure(self):
        self._tasksweep.configure()

    @property
    def task(self):
        return self._tasksweep.task

    def get_task_cidx(self, state):
        return self._tasksweep.get_cidxs(*state)

    @property
    def _squeeze(self):
        return self._tasksweep.squeeze


class ContainerNamespace:
    # read-only attributes except for underscore
    def __setattr__(self, name, value):
        if not name.startswith('_'):
            msg = "can't set attribute {!r}".format(name)
            raise AttributeError(msg)
        object.__setattr__(self, name, value)

    def __init__(self):
        self._params = OrderedDict()

    def _append(self, name, param):
        if name in self._params:
            self._params.pop(name)
        object.__setattr__(self, name, param)
        self._params[name] = param

    def __iter__(self):
        for name, param in self._params.items():
            yield name, param

    def __getitem__(self, name):
        return self._params[name]

    def __contains__(self, value):
        if isinstance(value, str):
            return value in self._params.keys()
        else:
            return value in self._params.values()

    def __len__(self):
        return len(self._params)

    def __repr__(self):
        params = [repr(p).strip('<>') for p in self._params.values()]
        return '<[{}]>'.format(', '.join(params))


class Parameters(ContainerNamespace):
    def __init__(self, task):
        self._task = task
        super().__init__()

    @property
    def _idxs(self):
        return tuple(param.idx for name, param in self)

    def _get(self, value):
        if isinstance(value, (tuple, list)):
            params = []
            for item in value:
                params += self._get(item)
            return params
        else:
            return self.__get(value)

    def _get_paths(self, value):
        if isinstance(value, (tuple, list)):
            params = []
            for item in value:
                params.append(self._get(item))
            return params
        else:
            return [self.__get(value)]

    def __get(self, value):
        dep_tasks = self._task.depend_tasks.keys()
        if value in ('', None):
            return []
        elif isinstance(value, str):
            return [self[name.strip()] for name in value.split(',')]
        elif value in self._params.values():
            return [value]
        elif value in dep_tasks:
            return [value]
        elif value is self._task:
            return [value]
        else:
            try:
                if value._task in dep_tasks:
                    return [value]
                else:
                    raise KeyError
            except (AttributeError, KeyError):
                msg = '{!r} is not a parameter of task {}'
                msg = msg.format(value, self._task)
                raise ValueError(msg)


class NumpyPrettyPrint():
    """from
    https://stackoverflow.com/questions/38050643/local-scope-for-numpy-set-printoptions
    """
    def __init__(self, **options):
        self.options = options

    def __enter__(self):
        self.back = np.get_printoptions()
        np.set_printoptions(**self.options)

    def __exit__(self, *args):
        np.set_printoptions(**self.back)


class ArgumentParams(Parameters):
    def __init__(self, task=None):
        super().__init__(task)
        self._nested = Nested()
        self._nested_args = {}
        self._tasksweeps = {}           # task: tasksweep
        self._tasksweep_args = {}       # arg:  tasksweep

        # used as cache for configuration
        self._last_tasksweep = None
        self._last_tasksweep_args = []  # [arg, ...]

    def __repr__(self):
        maxlen = max([0] + [len(param._uname) for name, param in self])
        fmtstr = '{{:>{maxlen}}} = '.format(maxlen=maxlen)
        lines = []
        for name, param in self:
            line = fmtstr.format(param._uname)
            val_str = _value_str(param.value).split('\n')
            gap = '\n' + ' ' * len(line)
            val_str = gap.join(val_str)
            line += '{}'.format(val_str)
            ptr = param._ptr
            if isinstance(ptr, (SweepPointer, SweepFactoryPointer)):
                if len(param._ptrs._pointers) == 1:
                    sweep = ptr.sweep
                    ptr_repr = sweep
                else:
                    sweep = param._ptrs
                    ptr_repr = [ptr.sweep for ptr in sweep._pointers]
                line += '\t({}/{}):\t{}'.format(sweep.idx + 1, len(sweep),
                                                ptr_repr)
            elif isinstance(ptr, DependParamPointer):
                sweep = ptr.sweep
                ptr_repr = 'from {!r}'.format(ptr._param)
                squeeze = ptr._tasksweep.squeeze
                if squeeze:
                    squeeze_str = ', '.join(repr(path) for path in squeeze)
                    ptr_repr += ', squeeze {}'.format(squeeze_str)
                line += '\t({}/{}):\t{}'.format(sweep.idx + 1, len(sweep),
                                                ptr_repr)
            lines.append(line)
        return '\n'.join(lines)

    def zip(self, *items):
        if len(items) < 2:
            msg = 'need at least two items in order to zip'
            raise ValueError(msg)
        args = []
        for item in items:
            arg = self[item] if isinstance(item, str) else item
            args.append(arg)
        idx = self._nested_args[args[0]]
        for arg in args[1:]:
            self._nested_args[arg] = idx

    def _get_zipped_args(self, value):
        try:
            levels = self._nested_levels
            all_args = set()
            for arg in self._get(value):
                if arg in self._nested_args:
                    level = self._nested_args[arg]
                    all_args.update(levels[level])
            return all_args
        except ValueError:
            return value._task.args._get_zipped_args(value)

    @property
    def _nested_levels(self):
        """inverse of self._nested_args"""
        levels = {}
        for arg, level in self._nested_args.items():
            levels.setdefault(level, []).append(arg)
        return levels

    def _get_non_squeezed_args(self, sq_args):
        dep_tasks = set()
        args = []
        for name, arg in self:
            if arg in sq_args:
                continue
            tasks = arg._tasks
            if tasks:
                dep_tasks.update(tasks)
            else:
                # local arg
                args.append(arg)
        for task in dep_tasks:
            args.extend(task.args._get_non_squeezed_args(sq_args))
        return args

    def _get_depending_args(self, sq_paths=None, pre_tasks=None, all=0):
        if sq_paths is None:
            sq_paths = []
        if pre_tasks is None:
            pre_tasks = []
        dep_tasks = set()
        paths = []
        #~ print('*** {} ***'.format(self._task))
        #~ print('sq_paths:', sq_paths)
        for name, arg in self:
            arg_path = pre_tasks + [arg]
            #~ print('arg_path:', arg_path)
            if arg_path in sq_paths:
                sq_paths.remove(arg_path)
                continue
            tasks = arg._tasks
            if tasks:
                dep_tasks.update(tasks)
                if all:
                    paths.append([arg])
            else:
                # local arg
                paths.append([arg])
        pre_tasks = list(pre_tasks)
        pre_tasks.append(self._task)
        for task in dep_tasks:
            dep_paths = task.args._get_depending_args(
                sq_paths,
                pre_tasks,
                all=all,
            )
            for path in dep_paths:
                path.insert(0, self._task)
            paths += dep_paths
        return paths

    def _get_acc_paths(self, squeeze=''):
        squeeze = self._get(squeeze)
        squeeze = self._zipped_paths(squeeze)
        sq_paths = self._guess_paths(squeeze)
        # transform tasks in path into proper args
        sq_paths = self._task_paths_to_arg_paths(sq_paths)
        return sq_paths

    def _get_key_paths(self, squeeze=''):
        squeeze = self._get(squeeze)
        squeeze = self._zipped_paths(squeeze)
        sq_paths = self._guess_paths(squeeze)
        key_paths = self._get_depending_args(list(sq_paths))
        # transform tasks in path into proper args
        key_paths = self._task_paths_to_arg_paths(key_paths)
        sq_paths = self._task_paths_to_arg_paths(sq_paths)
        return key_paths, sq_paths

    def _guess_paths(self, squeeze):
        arg_paths = sorted(self._get_depending_args(all=1), key=len)
        sq_paths = []
        for sq_path in squeeze:
            #~ print('sq_path:', sq_path)
            for arg_path in arg_paths:
                #~ print('arg_path:', arg_path)
                if all(item in arg_path for item in sq_path):
                    sq_paths.append(arg_path)
                    break
        return list(sq_paths)

    @staticmethod
    def _zipped_paths(task_path):
        task_path = list(task_path)
        task_path_with_zipped_args = []
        for sq_arg in task_path:
            if isinstance(sq_arg, Argument):
                task_path.remove(sq_arg)
                task_path_with_zipped_args = []
                for arg in sq_arg._task.args._get_zipped_args(sq_arg):
                    task_path_with_zipped_args.append([arg] + task_path)
                break
        return task_path_with_zipped_args

    @staticmethod
    def _task_paths_to_arg_paths(task_paths):
        arg_paths = []
        for path in task_paths:
            arg = path[-1]
            arg_path = [arg]
            for task in path[-2::-1]:
                #~ print('   ', arg, task)
                arg = task.depend_tasks[arg._task][0]
                arg_path.insert(0, arg)
            arg_paths.append(arg_path)
        return arg_paths

    def _add_sweeped_arg(self, arg):
        idx = max([0] + list(self._nested_args.values()))
        self._nested_args[arg] = idx + 1
        #~ self._tasksweeps.pop(arg, None)

    def _remove_sweeped_arg(self, arg):
        self._nested_args.pop(arg, None)

    def _configure(self):
        for name, arg in self:
            arg.configure()
        if self._has_changed():
            nested = {}
            for arg, idx in self._nested_args.items():
                args = nested.setdefault(idx, [])
                args.append(arg)
            sweeps = []
            for idx in sorted(nested):
                args = nested[idx]
                if len(args) == 1:
                    sweep = args[0]._ptrs
                else:
                    sweep = Zip(*[arg._ptrs for arg in args])
                sweeps.append(sweep)
            self._nested.sweeps = sweeps
            self._nested._is_initialized = True

    def _has_changed(self):
        changed = []
        for name, arg in self:
            value = not len(arg._cache) or arg._cache[-1] != arg.state
            changed.append(value)
        return any(changed)

    def _get_key_dict(self, cidx):
        keys = {}
        for name, arg in self:
            arg_state = arg._cache[cidx]
            iptr, state = arg_state
            ptr = arg._ptrs._pointers[iptr]
            try:
                # ToDo: prepend iptr to the depending arg-states
                keys.update(ptr._get_key_dict(state))
            except AttributeError:
                keys[arg] = arg_state
        return keys

    def _reset(self):
        self._nested.reset()

    def _is_running(self):
        return self._nested.is_running()

    def _is_finished(self):
        return not self._is_running()

    def as_table(self, cidx=None, include=[], hide_const=False,
                 as_df=True):
        if isinstance(cidx, int):
            cidx = [cidx]
        params = OrderedDict()
        for name, arg in self:
            if not hide_const or len(arg._states) > 1:
                params[arg._uname] = arg.get_cache(cidx)
        for param in include:
            if isinstance(param, (list, tuple)):
                param, via = param[0], param[1:]
            else:
                via = []
            path = self._task.get_path(param, via)
            values = self._task.get_value_from_path(path, cidx)
            name = '{}.{}.{}'.format(
                param._task.name,
                'args' if isinstance(param, Argument) else 'results',
                param._uname
            )
            params[name] = values
        if cidx:
            params['cidx'] = list(cidx)
        if not as_df:
            return params
        df = pd.DataFrame(params)
        if cidx:
            df = df.set_index('cidx')
            df.index.name = None
        return df

    def select_by_sweep(self, sweeps=[], squeeze=[], cidx=None,
                        deps=None, **kwargs):
        """Returns selected cache indices by sweeped argument names.

        Additionally returns sweeped argument states.


        >>> def func(a, b, c, d, e=5):
        ...     return
        >>> task = Task(func)
        >>> task.args.a.sweep(0, 2)
        >>> task.args.b.sweep(10, 12)
        >>> task.args.c.sweep(100, 102)
        >>> task.args.d.sweep(500, 600, step=100)
        >>> task.run(2)
        >>> task.run(1)
        >>> task.run(0)
        >>> idxs, args = task.args.select_by_sweep(sweeps='b', squeeze='a')
        >>> idxs
        [[9, 10, 11], [12]]
        >>> args
        [((0, 0),), ((0, 1),)]
        >>> task.run()
        >>> len(task.args._nested)
        54
        >>> idxs, _ = task.args.select_by_sweep()
        >>> idxs
        [53]
        >>> idxs, _ = task.args.select_by_sweep(cidx=32)
        >>> idxs
        [32]
        >>> idxs, _ = task.args.select_by_sweep(squeeze=['d', 'c'])
        >>> idxs
        [[8, 35, 17, 44, 26, 53]]
        >>> task.args.as_table(idxs[0])
            a   b    c    d  e
        8   2  12  100  500  5
        35  2  12  100  600  5
        17  2  12  101  500  5
        44  2  12  101  600  5
        26  2  12  102  500  5
        53  2  12  102  600  5
        >>> idxs, args = task.args.select_by_sweep(['b', 'a'], ['d', 'c'])
        >>> idxs
        [[0, 27, 9, 36, 18, 45],
         [3, 30, 12, 39, 21, 48],
         [6, 33, 15, 42, 24, 51],
         [1, 28, 10, 37, 19, 46],
         [4, 31, 13, 40, 22, 49],
         [7, 34, 16, 43, 25, 52],
         [2, 29, 11, 38, 20, 47],
         [5, 32, 14, 41, 23, 50],
         [8, 35, 17, 44, 26, 53]]
        >>> args
         [((0, 0), (0, 0)),
          ((0, 1), (0, 0)),
          ((0, 2), (0, 0)),
          ((0, 0), (0, 1)),
          ((0, 1), (0, 1)),
          ((0, 2), (0, 1)),
          ((0, 0), (0, 2)),
          ((0, 1), (0, 2)),
          ((0, 2), (0, 2))]
        >>> idxs, _ = task.args.select_by_sweep(['b', 'a'])
        >>> idxs
        [45, 48, 51, 46, 49, 52, 47, 50, 53]
        >>> task.args.as_table(idxs)
            a   b    c    d  e
        45  0  10  102  600  5
        48  0  11  102  600  5
        51  0  12  102  600  5
        46  1  10  102  600  5
        49  1  11  102  600  5
        52  1  12  102  600  5
        47  2  10  102  600  5
        50  2  11  102  600  5
        53  2  12  102  600  5
        >>> idxs, _ = task.args.select_by_sweep(['b', 'a'], cidx=50)
        >>> idxs
        [45, 48, 46, 49, 47, 50]
        >>> idxs,_=task.args.select_by_sweep(['b', 'a'], c=(0, 0), cidx=32)
        >>> idxs
        [27, 30, 28, 31, 29, 32]
        >>> task.args.as_table(idxs)
            a   b    c    d  e
        27  0  10  100  600  5
        30  0  11  100  600  5
        28  1  10  100  600  5
        31  1  11  100  600  5
        29  2  10  100  600  5
        32  2  11  100  600  5
        """
        if isinstance(sweeps, str):
            sweeps = sweeps.split(',')
        if isinstance(squeeze, str):
            squeeze = squeeze.split(',')
        sweeps = [name.strip() for name in sweeps]
        squeeze = [name.strip() for name in squeeze]
        swargs = [self[name] for name in sweeps]
        sqargs = [self[name] for name in squeeze]
        other = set(self._params.values())
        other = other.difference(swargs)
        #~ other = other.difference(sqargs)
        other = other.difference(kwargs.keys())
        if deps is None:
            deps = self._task.get_depend_args()
        for name in itertools.chain(sweeps, squeeze,kwargs.keys()):
            if name in deps:
                for arg in deps[name]:
                    if arg in other:
                        other.remove(arg)
        other_args = {}
        for arg in other:
            if not arg._cache:
                return [], []
            state = arg._cache[-1] if cidx is None else arg._cache[cidx]
            other_args[arg.name] = state
        other_args.update(kwargs)
        oidxs = self.select_by_idx(**other_args)
        # maybe not necessary but I am carfully
        if cidx is not None:
            oidxs = set(idx for idx in oidxs if idx <= cidx)

        idxs = []
        arg_states = []
        arg_idx_sets = [sorted(arg._states.keys()) for arg in swargs]
        for aid in itertools.product(*reversed(arg_idx_sets)):
            if squeeze:
                args_idx = {n: idx for n, idx in zip(sweeps, aid[::-1])}
                args_idx.update(kwargs)
                iset, _ = self.select_by_sweep(squeeze, cidx=cidx,
                                               deps=deps, **args_idx)
                if iset:
                    idxs.append(iset)
                    arg_states.append(aid[::-1])
            else:
                i = [a._states[idx] for a, idx in zip(swargs, aid[::-1])]
                idx_set = set.intersection(oidxs, *i)
                if cidx is not None:
                    idx_set = set(idx for idx in idx_set if idx <= cidx)
                if idx_set:
                    idxs.extend(sorted(idx_set))
                    arg_states.append(aid[::-1])
        return idxs, arg_states

    def select_by_idx(self, **kwargs):
        names = OrderedDict()
        for name, state in kwargs.items():
            if name in self:
                arg = self[name]
                names[name] = arg._states[state]
        idxs = set(range(self._task.clen))
        idxs = idxs.intersection(*names.values())
        return idxs

    def select_by_value(self, **kwargs):
        names = OrderedDict()
        for name, value in kwargs.items():
            if name in self:
                arg = self[name]
                values = np.asarray(arg.get_cache())
                cidx = np.abs(values - value).argmin()
                idx = arg._cache[cidx]
                names[name] = idx
        idxs = self.select_by_idx(**names)
        return self.as_table(sorted(idxs))


class ReturnParams(Parameters):
    def __repr__(self):
        maxlen = max([0] + [len(param._uname) for name, param in self])
        fmtstr = '{{:>{maxlen}}} = {{}}'.format(maxlen=maxlen)
        lines = []
        for name, param in self:
            line = fmtstr.format(param._uname, _value_str(param.value))
            lines.append(line)
        return '\n'.join(lines)

    def as_table(self, include_args=True, include=[], hide_const=False,
                 as_df=True, cidx=None):
        if isinstance(cidx, int):
            cidx = [cidx]
        params = OrderedDict()

        if include_args:
            for name, arg in self._task.args:
                arg_keys = arg._states.keys()
                if (not hide_const
                    or len(arg_keys) > 1
                    or self._task.clen < 2
                ):
                    params[arg._uname] = arg.get_cache(cidx)
        for name, result in self._task.returns:
            params[result._uname] = result.get_cache(cidx)
        for param in include:
            if isinstance(param, (list, tuple)):
                param, via = param[0], param[1:]
                via = self._task.args._get(via)
            else:
                via = []
            path = self._task.get_path(param, via)
            values = self._task.get_value_from_path(path, cidx)
            name = '{}.{}.{}'.format(
                param._task.name,
                'args' if isinstance(param, Argument) else 'results',
                param._uname
            )
            params[name] = values
        if cidx:
            params['cidx'] = list(cidx)
        if not as_df:
            return params
        df = pd.DataFrame(params)
        if cidx:
            df = df.set_index('cidx')
            df.index.name = None
        return df


class Slice(object):
    def __init__(self, start=None, stop=None, step=None):
        self.start = start
        self.stop = stop
        self.step = step

    def __contains__(self, idx):
        is_start = (not self.start or self.start <= idx)
        is_stop = (not self.stop or idx <= self.stop)
        idx_start = idx if self.start is None else idx - self.start
        is_step = not (self.step and idx_start % self.step)
        return is_start and is_stop and is_step

    def __repr__(self):
        args = []
        args.append(repr(self.start))
        args.append(repr(self.stop))
        args.append(repr(self.step))
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

class IndexSet:
    def __init__(self, idx_set=None):
        if isinstance(idx_set, int):
            idx_set = {idx_set}
        elif idx_set is None:
            idx_set = Slice()
        self.idx_set = idx_set

    def __contains__(self, idx):
        return idx in self.idx_set

    def __repr__(self):
        args = []
        args.append(repr(self.idx_set))
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))


class EventManager:
    def __init__(self):
        self.events = []

    def next(self):
        for event, action in self.events:
            if event() and action is not None:
                action()

    def add(self, event, action=None):
        self.events.insert(0, (event, action))

    def add_idx_event(self, obj, idx, action=None):
        idx_set = IndexSet(idx)

        def event():
            value = obj.idx in idx_set
            if value:
                msg = '{}.idx = {}'.format(
                    type(obj),
                    obj.idx,
                )
                print(msg)
            return value

        if action is None:
            def action():
                try:
                    msg = 'event from {!r}: idx = {} is in {}'
                    msg = msg.format(obj, obj.idx, idx_set)
                except AttributeError:
                    msg = 'action'
                print(msg)
        self.add(event, action)


def return_values(func):
    names = []
    tree = ast.parse(dedent(inspect.getsource(func)))
    for exp in ast.walk(tree):
        if isinstance(exp, ast.FunctionDef):
            break
    annovars = {}
    for e in exp.body:
        # for python3.6
        if isinstance(e, ast.AnnAssign):
            varname = e.target.id
            if isinstance(e.annotation, ast.Name):
                annovars[varname] = e.annotation.id
            elif isinstance(e.annotation, ast.Str):
                annovars[varname] = e.annotation.s
        if isinstance(e, ast.Return):
            break
    if not isinstance(e, ast.Return):
        return tuple()
    v = e.value
    if isinstance(v, (ast.Tuple, ast.List)):
        for idx, item in enumerate(v.elts):
            if isinstance(item, ast.Name):
                names.append(item.id)
            else:
                names.append('out_' + str(idx))
    elif isinstance(v, ast.Dict):
        for idx, item in enumerate(v.keys):
            if isinstance(item, ast.Str):
                names.append(item.s)
            else:
                names.append('out_' + str(idx))
    elif isinstance(v, ast.Call) and v.func.id == 'dict' and v.keywords:
        for idx, item in enumerate(v.keywords):
            names.append(item.arg)
    elif isinstance(v, ast.Name):
        names.append(v.id)
    elif v is not None:
        names.append('out')
    return tuple(names), annovars


def namedresult(func):
    result = namedtuple('result' , return_values(func))
    @wraps(func)
    def myfunc(*args, **kwargs):
        return result(*func(*args, **kwargs))
    return myfunc


class Task(BaseSweepIterator):
    """Task object for configured function func.


    >>> def func(a, b, c, d=3):
    ...     return
    >>> task = Task(func)

    >>> task.args.a.iterate(5, 10)
    >>> task.args.b.iterate(10, 20)
    >>> task.args.b.iterate(50, 200, concat=True)
    >>> task.args
    a = 5       (1/2):  Iterate(5, 10)
    b = 10      (1/4):  [Iterate(10, 20), Iterate(50, 200)]
    c = None
    d = 3

    >>> task.run()
    >>> task.args.as_table()
        a    b     c  d
    0   5   10  None  3
    1  10   10  None  3
    2   5   20  None  3
    3  10   20  None  3
    4   5   50  None  3
    5  10   50  None  3
    6   5  200  None  3
    7  10  200  None  3
    """
    def __init__(self, func, name=None, _tm=None):
        self.func = func
        self.name = func.__name__ if name is None else name
        if sys.version_info.major < 3:
            sig = inspect.getargspec(func)
            defaults = dict(zip_longest(sig.args[::-1], sig.defaults[::-1]))
        else:
            defaults = OrderedDict()
            for name, param in inspect.signature(func).parameters.items():
                if (param.kind is not inspect.Parameter.VAR_POSITIONAL and
                    param.kind is not inspect.Parameter.VAR_KEYWORD):
                    if param.default is param.empty:
                        defaults[name] = None
                    else:
                        defaults[name] = param.default

        args = ArgumentParams(task=self)
        for n, d in defaults.items():
            unit = func.__annotations__.get(n, None)
            args._append(n, Argument(n, d, unit, task=self))
        self.args = args

        returns = ReturnParams(task=self)
        names, annovars = return_values(func)
        for name in names:
            unit = annovars.get(name, None)
            returns._append(name, ReturnValue(name, unit, self))
        self.returns = returns

        params = Parameters(task=self)
        for name, arg in args:
            params._append(name, arg)
        for name, result in returns:
            params._append(name, result)
        self.params = params

        self.clen = 0

        super().__init__()

        self.em = EventManager()
        self.am = AxesManager()

        # new plotmanager
        self._pm = None
        log.debug('created task: {}'.format(self))

        self._config = []
        self._tm = _tm

        self._loglevel = None

    def reset(self):
        self.args._reset()
        self.clen = 0
        for name, param in self.params:
            param._reset()
        if self._pm:
            self._pm._reset()

    def __repr__(self):
        return '<{}>'.format(self.name)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __len__(self):
        return self.args._nested.__len__()

    @property
    def idx(self):
        return self.args._nested.idx

    @idx.setter
    def idx(self, value):
        self.args._nested.idx = value

    def next_idx(self):
        self.args._nested.next_idx()
        return self.args._nested.idx

    def _call(self):
        kwargs = {}             # name: value   for self.func(**kwargs)
        sweeped_lines = {}      # level: line   for logging
        const_lines = []        # line          for logging
        levels = sorted(self.args._nested_args.values())
        loglevel = self._loglevel
        for name, arg in self.args:
            value = arg.value
            kwargs[name] = value
            arg._capture_idx()
            # logging
            ptr = arg._ptr
            if arg in self.args._nested_args and loglevel is not None:
                level = self.args._nested_args[arg]
                sweep = ptr.sweep
                if loglevel and levels.index(level) < loglevel:
                    head = '...: '
                    line = arg.name
                else:
                    head = '{}/{}: '.format(sweep.idx + 1, len(sweep))
                    line = '{} = {}'.format(arg._uname, _value_str(value))
                sweeped_lines[level] = head, line
            elif loglevel is not None:
                line = '{} = {}'
                line = line.format(arg._uname, _value_str(value))
                if isinstance(ptr, DependParamPointer):
                    line += ':  from {!r}'.format(ptr._param)
                    t = ptr._param._task
                    if ptr._squeeze:
                        _params = [repr(t.params[p]) for p in ptr._squeeze]
                        _params = ', '.join(_params)
                        line += ', squeeze={}'.format(_params)
                const_lines.append(line)
        lines = []
        hmax = 8
        if sweeped_lines:
            lines.append('    looping args:')
            indent = 8
            hmax = max(len(head) for head, line in sweeped_lines.values())
            for nlev in reversed(sorted(sweeped_lines)):
                head, line = sweeped_lines[nlev]
                hline = '{{:>{}}}'.format(hmax + indent)
                #~ print('{!r}, {!r}'.format(hline, head))
                hline = hline.format(head)
                hline += line
                lines.append(hline)
        if const_lines:
            lines.append('    constant args:')
            for line in const_lines:
                lines.append(' ' * (0*hmax + indent) + line)
        if lines:
            log.info('\n'.join(lines))
            self._loglevel = None
        _result = self.func(**kwargs)

        self.clen += 1
        result = _result

        if not isinstance(result, (tuple, list, dict, OrderedDict)):
            result = [result]

        if isinstance(result, (tuple, list)):
            for (name, param), value in zip(self.returns, result):
                param.value = value
        elif isinstance(result, (dict, OrderedDict)):
            for name, param in self.returns:
                param.value = result[name]

        return_lines = []
        cidx = self.clen - 1
        for name, retval in self.returns:
            value = retval.get_cache(cidx)
            line = '        {} = {}'.format(retval._uname,
                                            _value_str(value))
            return_lines.append(line)
        if return_lines and loglevel == 0:
            log.info('    return values:')
            log.info('\n'.join(return_lines))

        return _result

    def call(self):
        nested = self.args._nested
        _result = self._call()
        return _result

    def out(self, index=None):
        res = self.returns
        if index is None:
            return {n: p.value for n, p in res}
        elif isinstance(index, (int, slice)):
            return {n: p._cache[index] for n, p in res}
        elif isinstance(index, (tuple, list)):
            values = {}
            for n, p in res:
                values[n] = [p._results[idx] for idx in index]
            return values
        else:
            raise ValueError('index must be either None, slice '
                             'or tuple/list')

    def __getitem__(self, idx):
        return self.out(idx)

    def next_value(self):
        self.args._nested.next_value()
        self.call()
        self.em.next()
        return self.value

    def run(self, num=None, inner=False, plot_update=False):
        """run the loops of the task

        task.run()           run until task is finished

        task.run(0)          one single loop step

        task.run(1)          next value of second loop until
                             inner loops are finished

        task.run(1, True)    next value of second loop until
                             the same inner loop idx
        """
        self.args._configure()
        plevel = plotmanager.log.level
        plotmanager.log.setLevel('WARNING')
        if plot_update and self.args._nested.is_running():
            self.pm.open_window(0)
        sweeps = self.args._nested.sweeps[:num]
        inner_pre = [s.idx for s in sweeps]
        nested = self.args._nested
        if num != 0:
            line = 'run {!r} from'.format(self)
            line_len = len(line)
            nlen = str(len(nested))
            fmtstr = ' {{:{}}}/{}'.format(len(nlen), nlen)
            line += fmtstr + ' ...'
        else:
            line = 'run {!r}    {{}}/{}'
            line = line.format(self,    len(nested))
        if self.is_finished():
            line = ''
        else:
            log.info(line.format(nested.idx_next + 1))
        self._loglevel = num
        while 1:
            try:
                self.next_value()
                if plot_update:
                    self.plot_update()
            except StopIteration:
                break
            sweeps = self.args._nested.sweeps[:num]
            if all([s.is_finished() for s in sweeps]):
                break
        if inner:
            #~ self._loglevel = num
            sweeps = self.args._nested.sweeps[:num]
            while self.is_running() and \
                  [s.idx for s in sweeps] != inner_pre:
                self.next_value()
                if plot_update:
                    self.plot_update()
        if num != 0 and line:
            line = ' to'
            line = '.' * (line_len - len(line)) + line
            line += fmtstr.format(self.args._nested.idx + 1)
            log.info(line)
        if not plot_update:
            self.plot_update()
        plotmanager.log.level = plevel

    def add_dependency(self, task, squeeze=''):
        if task not in self.args._tasksweeps:
            tasksweep = TaskSweep(task, squeeze)
            self.args._tasksweeps[task] = tasksweep
        self.args._last_tasksweep = self.args._tasksweeps[task]
        self.args._last_tasksweep_args = []

    @property
    def depend_tasks(self):
        """Returns a map from depending tasks to its needed arguments

        >>> @Task
        ... def one(x):
        ...     y = x + 1
        ...     return y
        >>> one.args.x.sweep(0, 2)

        >>> @Task
        ... def two(x, offs=0):
        ...     y = x**2 + offs
        ...     return y
        >>> two.args.x.depends_on(one.returns.y, squeeze='x')
        >>> two.args.offs.iterate(0, 10)

        >>> @Task
        ... def three(x1, x2, x3):
        ...     y = x1 * sum(x2) + x3
        ...     return y

        #~ >>> three.args.x1.depends_on(two.returns.y, squeeze='offs')
        >>> three.args.x1.depends_on(two.args.offs, squeeze='offs')
        >>> three.args.x2.depends_on(one.returns.y, squeeze='x')
        >>> three.args.x3.depends_on(one.returns.y)

        >>> @Task
        ... def four(x1, x2=0):
        ...     y = x1 + x2
        ...     return y
        >>> four.args.x1.depends_on(three.returns.y)
        >>> four.args.x2.sweep(1, 3)

        >>> four.depend_tasks
        OrderedDict([(<three>, [<four.args.x1>]),
                     (<two>, [<three.args.x1>]),
                     (<one>, [<three.args.x2>,
                              <three.args.x3>,
                              <two.args.x>])])

        >>> four.get_path(two.args.x)
        [<four.args.x1>, <three.args.x1>, <two.args.x>]

        >>> four.get_path(one.args.x, [three.args.x2])
        [<four.args.x1>, <three.args.x2>, <one.args.x>]

        >>> four.get_path(one.args.x, via_args=[two.args.x])
        [<four.args.x1>, <three.args.x1>, <two.args.x>, <one.args.x>]

        >>> one.run()
        >>> one.returns.as_table()
           x  y
        0  0  1
        1  1  2
        2  2  3

        >>> two.run()
        >>> two.returns.as_table()
                   x  offs             y
        0  [1, 2, 3]     0     [1, 4, 9]
        1  [1, 2, 3]    10  [11, 14, 19]

        >>> three.run()
        >>> three.returns.as_table()
                x1         x2  x3        y
        0  [0, 10]  [1, 2, 3]   3  [3, 63]

        >>> four.run()
        >>> four.returns.as_table()
                x1  x2        y
        0  [3, 63]   1  [4, 64]
        1  [3, 63]   2  [5, 65]
        2  [3, 63]   3  [6, 66]

        >>> four.get_value_from_path([three.args.x1], cidx=2)
        array([ 0, 10])

        >>> four.get_value_from_path([three.args.x2], cidx=2)
        array([1, 2, 3])

        >>> four.get_value_from_path([two.args.offs], cidx=2)
        [0, 10]

        >>> four.get_value_from_path([two.returns.y], cidx=2)
        [array([1, 4, 9]), array([11, 14, 19])]

        >>> four.get_value_from_path([two.args.x], cidx=2)
        [array([1, 2, 3]), array([1, 2, 3])]

        >>> path = [(one.args.x, two.args.x)]
        >>> four.get_value_from_path(path, cidx=[1, 2])
        [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]]
        """
        tasks = OrderedDict()
        for name, arg in self.args:
            for task in arg._tasks:
                tasks.setdefault(task, []).append(arg)
        for task in list(tasks):
            for tsk, newargs in task.depend_tasks.items():
                args = tasks.setdefault(tsk, [])
                args += [narg for narg in newargs if narg not in args]
        return tasks

    def get_path(self, param, via_args=[]):
        via = set(via_args)
        tasks = self.depend_tasks
        params = [param]
        while 1:
            task = params[0]._task
            if task is self:
                break
            else:
                args = tasks[task]
                if len(args) > 1:
                    try:
                        arg = via.intersection(args).pop()
                    except KeyError:
                        # workaround:
                        # try to avoid via_args if multiple arguments
                        # depends in the same way on an other task
                        # => check if these args have the same nested
                        # levels
                        levels = {}
                        for _arg in args:
                            _task = _arg._task
                            level = _task.args._nested_args[_arg]
                            levels.setdefault(_task, set()).add(level)
                        if len(levels) == 1 and len(levels[_task]) == 1:
                            arg = args[0]
                        else:
                            msg = 'no hint in via_args for depending ' \
                                  'arguments {} of task {}'
                            raise ValueError(msg.format(args, task))
                else:
                    arg = args[0]
                params.insert(0, arg)
        return params

    def _get_argpath(self, value):
        if value in ('', None):
            return []
        elif isinstance(value, str):
            return [self.params[value]]
        elif isinstance(value, (tuple, list)):
            args = []
            for item in value:
                if isinstance(item, str):
                    args.append(self.params[item])
                else:
                    args.append(item)
            param, via = args[0], args[1:]
            return self.get_path(param, via)
        else:
            return self.get_path(value)

    def get_cidx_from_path(self, path, cidx=None):
        if cidx is None:
            cidx = range(self.clen)
        try:
            return [self.get_cidx_from_path(path, idx) for idx in cidx]
        except TypeError:
            task = self
            for arg in path[:-1]:
                if task == arg._task:
                    task = arg._ptr.task
                else:
                    msg = 'task {} != {}'.format(task, arg._task)
                    raise ValueError(msg)
                cidx = arg.get_depend_cidx(cidx)
                if isinstance(cidx, list):
                    cidx = cidx[0]
        return cidx

    # ToDo: should be in self.args
    def get_value_from_path(self, path, cidx=None):
        if cidx is None:
            cidx = range(self.clen)
        try:
            return [self.get_value_from_path(path, idx) for idx in cidx]
        except TypeError:
            task = self
            for arg in path[:-1]:
                if task == arg._task:
                    task = arg._ptr.task
                else:
                    msg = 'task {} != {}'.format(task, arg._task)
                    raise ValueError(msg)
                cidx = arg.get_depend_cidx(cidx)
                # guess:
                #   if cidx is list, then arg has a squeezed dependency
                # ToDo:
                #   fold cidx if next arg is in squeezed arg._task.args
                #   but is NOT the squeezed arg itself
            if isinstance(cidx, list):
                for sq_arg in arg._get_squeezed_args():
                    if (path[-1] in sq_arg._task.args
                        and path[-1] is not sq_arg
                    ):
                        cidx = cidx[0]
            return path[-1].get_cache(cidx)

    # ToDo: should be in task.args
    def get_depend_args(self):
        params = {}
        for name, arg in self.args:
            if isinstance(arg._ptr, DependParamPointer):
                args = params.setdefault(arg._ptr._param, [])
                args.append(arg)
        levels = self.args._nested_levels
        dep_args = {}
        for name, arg in self.args:
            if arg in self.args._nested_args:
                idx = self.args._nested_args[arg]
                args = levels[idx]
                if len(args) > 1:
                    dep_args[name] = [a for a in args if a is not arg]
            elif isinstance(arg._ptr, DependParamPointer):
                args = params[arg._ptr._param]
                if len(args) > 1:
                    dep_args[name] = [a for a in args if a is not arg]
        return dep_args

    def plot_update(self):
        if self._pm is not None:
            self.pm.update(self)

    @config
    def plot(self, x='', y='', squeeze=None, accumulate=None,
             row=0, col=0, use_cursor=True, **kwargs):
        self.pm.plot(self, x, y, squeeze, accumulate, row, col,
                     use_cursor, **kwargs)
        self.plot_update()

    @property
    def pm(self):
        if self._tm is None:
            if self._pm is None:
                plevel = plotmanager.log.level
                plotmanager.log.setLevel('WARNING')
                self._pm = plotmanager.PlotManager()
                plotmanager.log.level = plevel
        else:
            self._pm = self._tm.pm
        return self._pm

    def _plot(self, x='', y='', sweeped_arg='', row=0, col=0):
        try:
            xarg = self.params[x]
        except KeyError:
            xarg = x
        try:
            yarg = self.param[y]
        except KeyError:
            yarg = y
        pm = SweepPlotManager(xarg, yarg, am=self.am)
        self.am.append_plot_manager(pm, row, col)
        #~ self.em.add_idx_event(self.nested, idx=Slice(), action=pm.plot)
        if not sweeped_arg:
            swparg = xarg
        elif isinstance(sweeped_arg, str):
            swparg = self.args[sweeped_arg]
        else:
            msg = 'no sweeped arg available'
            raise ValueError(msg)

        args = list(self.nested_args)
        args.remove(swparg)
        print('args = {}'.format(args))
        idx_prod = itertools.product(*[range(len(arg)) for arg in args])
        for idxs in idx_prod:
            print('idxs = {}'.format(idxs))
            idx_set = IndexSet()
            def ev_func(idxs=idxs):
                values = [arg.idx == idx for arg, idx in zip(args, idxs)]
                retval = all(values) and swparg.idx in idx_set
                if retval:
                    values = []
                    for arg, idx in zip(args, idxs):
                        value = '{}.idx={}'.format(arg.name, idx)
                        values.append(value)
                    values = ', '.join(values)
                    msg = '    event: {}.idx={} ({})'
                    msg = msg.format(swparg.name, swparg.idx, values)
                    print(msg)
                return retval
            def action(idxs=idxs):
                kwargs = {arg.name: idx for arg, idx in zip(args, idxs)}
                aidxs = self.args.select_by_idx(**kwargs)
                xdata = xarg.get_cache(aidxs)
                ydata = yarg.get_cache(aidxs)
                msg = '    action: idxs = {}'.format(aidxs)
                msg += '\n            xdata = {}'.format(xdata)
                msg += '\n            ydata = {}'.format(ydata)
                print(msg)
            self.em.add(ev_func, action)

        # bug for
        # quad.plot('x', 'y', sweep='m', col=1)
        #~ self.em.add_idx_event(sweep, idx=0, action=pm.newline)

    def to_csv(self, filename=''):
        if not filename:
            filename = self.name + '.csv'
        file = open(filename, 'w')

        for line in TabWriter(self).lines():
            file.write('## {}\n'.format(line))

        if self.clen:
            header = []
            header += ['arg_' + param._uname for _, param in self.args]
            header += [param._uname for _, param in self.returns]
            header += ['state_' + name for name, param in self.args]
            file.write('; '.join(header) + '\n')
        for cidx in range(self.clen):
            line = []
            for name, param in self.args:
                line.append(self._encode(param.get_cache(cidx)))
            for name, param in self.returns:
                line.append(self._encode(param.get_cache(cidx)))
            for name, param in self.args:
                line.append(self._encode(param._cache[cidx]))
            file.write('; '.join(line))
            file.write('\n')
        file.close()

    @staticmethod
    def _encode(value):
        if isinstance(value, np.ndarray):
            dtype = value.dtype
            shape = 'x'.join(str(s) for s in value.shape)
            data = value.tolist()
            return repr('data:{},{},{}'.format(dtype, shape, data))
        else:
            return repr(value)

    def _to_dict(self):
        dct = OrderedDict()
        dct['__class__'] = self.__class__.__name__
        dct['name'] = self.name
        dct['func'] = '<some code in file...>'
        dct['defaults'] = {n: arg._default._value for n, arg in self.args}
        config = []
        for func, obj, args, kwargs in self._config:
            conf = OrderedDict()
            if isinstance(obj, Task):
                conf['cmd'] = func.__name__
            else:
                conf['cmd'] = 'args.{}.{}'.format(obj.name, func.__name__)
            if args:
                conf['args'] = args
            if kwargs:
                conf['kwargs'] = kwargs
            config.append(conf)
        dct['config'] = config
        return dct

    @classmethod
    def _read_csv(cls, filename):
        file = open(filename)
        jlines = []
        for line in file:
            if line.startswith('##'):
                jlines.append(line.strip('#'))
            else:
                break
        units = {}
        args = []
        returns = []
        for col in line.split('; '):
            name, _, unit = col.partition(' / ')
            if name.startswith('arg_'):
                _, __, name = name.partition('arg_')
                args.append(name)
                units[name] = unit
            elif not name.startswith('state_'):
                returns.append(name)
                units[name] = unit

        dct = json.loads('\n'.join(jlines))
        task = cls._from_dict(dct)

        # restore arg units
        for name, param in task.args:
            param._unit = units[name]

        # add return values
        for name in returns:
            unit = units[name]
            retval = ReturnValue(name, unit, task)
            task.returns._append(name, retval)
            task.params._append(name, retval)

        # read csv data into cache
        cidx = -1
        for cidx, line in enumerate(file):
            columns = line.strip('\n').split('; ')
            nargs = len(args)
            nreturns = len(returns)
            rcolumns = columns[nargs:nargs+nreturns]
            for param, col in zip(returns, rcolumns):
                value = ast.literal_eval(col)
                retvalue = task.returns[param]
                retvalue._cache.append(value)
            acolumns = columns[nargs+nreturns:]
            for param, col in zip(args, acolumns):
                value = ast.literal_eval(col)
                arg = task.args[param]
                arg._cache.append(value)
                arg._states.setdefault(value, set()).add(cidx)
        task.clen = cidx + 1
        # apply dct['config'] with task._apply_config(dct)
        return dct, task

    def _apply_config(self, dct, namespace={}):
        for conf in dct['config']:
            names = conf['cmd'].split('.')
            if len(names) == 1:
                func_name = names[0]
                func = getattr(self, func_name)
            else:
                param_namespace, param_name, func_name = names
                params = getattr(self, param_namespace)
                param = getattr(params, param_name)
                func = getattr(param, func_name)
            _args = []
            for arg in conf.get('args', []):
                try:
                    ptype = arg['__param__']
                    taskname = arg['task']
                    task = namespace[taskname]
                    if ptype == 'Argument':
                        _args.append(task.args[arg['name']])
                    else:
                        _args.append(task.returns[arg['name']])
                except (TypeError, KeyError):
                    _args.append(arg)
            _kwargs = conf.get('kwargs', {})
            func(*_args, **_kwargs)
        # state recover
        self.args._configure()
        self.args._nested._is_initialized = False
        for name, arg in self.args:
            if arg._cache:
                state = arg._cache[-1]
                arg.state = state

    @classmethod
    def read_csv(cls, filename):
        dct, task = cls._read_csv(filename)
        task._apply_config(dct)
        return task

    @classmethod
    def _from_dict(cls, dct):
        def func(*args, **kwargs):
            msg = 'function of task <{}> was not read from csv file'
            msg = msg.format(dct['name'])
            raise NotImplementedError(msg)
            return None
        func.__name__ = dct['name']
        task = cls(func)

        del task.returns.out
        del task.params.out
        task.returns._params.pop('out')
        task.params._params.pop('out')

        for name, default in dct['defaults'].items():
            arg = Argument(name, default, task=task)
            task.args._append(name, arg)
            task.params._append(name, arg)

        """
        for conf in dct['config']:
            names = conf['cmd'].split('.')
            if len(names) == 1:
                func = getattr(task, names[0])
            else:
                param = getattr(task.params, names[1])
                func = getattr(param, names[2])
            args = conf.get('args', ())
            kwargs = conf.get('kwargs', {})
            func(*args, **kwargs)
        """
        return task


class TabWriter:
    def __init__(self, obj):
        self.obj = obj
        self.reset()

    def reset(self):
        self.indents = [0]
        self.s = ''
        self.llen = 0

    def write(self, s):
        self.s += s
        self.llen += len(s)
        #~ self.indents[-1] += len(s)

    def newline(self):
        self.s += '\n' + ' ' * self.indents[-1]
        self.llen = 0

    def tab(self):
        self.indents.append(self.indents[-1] + self.llen)
        self.llen = 0

    def untab(self):
        self.indents.pop()

    def lines(self):
        return self.as_json().split('\n')

    def as_json(self):
        return self.to_json(self.obj)

    def to_json(self, o):
        """code from ...
        https://stackoverflow.com/questions/10097477/python-json-array-newlines
        """
        if isinstance(o, (str, bool, int, float)) or o is None:
            self.write(json.dumps(o))
        elif isinstance(o, dict):
            self.write("{")
            self.tab()
            for n, (k, v) in enumerate(o.items()):
                self.write('"' + str(k) + '": ')
                self.to_json(v)
                if n < len(o) - 1:
                    self.write(',')
                    self.newline()
            self.write("}")
            self.untab()
        elif isinstance(o, (list, tuple)):
            self.write("[")
            self.tab()
            for n, e in enumerate(o):
                self.to_json(e)
                if n < len(o) - 1:
                    self.write(',')
                    if isinstance(e, dict) or hasattr(e, '_to_dict'):
                        self.newline()
                    else:
                        self.write(' ')
            self.write("]")
            self.untab()
        elif isinstance(o, np.ndarray):
            self.to_json(o.flatten().tolist())
        else:
            try:
                self.to_json(o._to_dict())
            except AttributeError:
                self.to_json('{!r}'.format(o))
        return self.s


class TaskManager:
    def __init__(self, *tasks, name='tm', **kwargs):
        self.name = name

        self._config = []
        self.args = ArgumentParams(self)
        self._add_arg('task')
        self.args.task.iterate()
        self._config = []

        self.tasks = ContainerNamespace()
        for task in tasks:
            self.append(task)

        self._pm = None

        for name, default in kwargs.items():
            self.add_arg(name, default=default)

    def reset(self):
        self.args._reset()
        for name, param in self.args:
            param._reset()
        for n, task in self.tasks:
            task.reset()
        if self._pm:
            self._pm._reset()

    def __repr__(self):
        args = [repr(task)   for task in self.tasks._params.values()]
        return "<{classname} {name}: {args}>".format(
            classname=self.__class__.__name__,
            name=self.name,
            args=', '.join(args))

    @property
    def pm(self):
        if self._pm is None:
            plevel = plotmanager.log.level
            plotmanager.log.setLevel('WARNING')
            self._pm = plotmanager.PlotManager()
            plotmanager.log.level = plevel
        return self._pm

    def _append(self, func, name=None):
        if isinstance(func, Task):
            task = func
            task._tm = self
        else:
            name = self._get_name(func) if name is None else name
            task = Task(func, name=name, _tm=self)
        if self.name == task.name:
            msg = 'name conflict: task {!r} has the name of TaskManager'
            msg = msg.format(task.name)
            log.error(msg)
            return
        self.tasks._append(task.name, task)
        self.args.task._ptr.sweep.items = list(self.tasks._params.values())
        return task

    def append(self, func, name=None):
        task = self._append(func, name)
        check = self._check_global_args(task=task)
        if check:
            for name, arg in self.args:
                if name in task.args:
                    task.args[name].depends_on(arg)
        return task

    def _get_name(self, func):
        fname = func.__name__
        if fname not in self.tasks:
            return fname
        else:
            msg = 'name {!r} is already used in {}.tasks'
            msg = msg.format(fname, self.name)
            log.warning(msg)
            n = 2
            while 1:
                name = fname + '_{}'.format(n)
                if name not in self.tasks:
                    msg = 'use {!r} instead for function {!r}'
                    msg = msg.format(name, fname)
                    log.warning(msg)
                    return name
                else:
                    msg = 'name {!r} is already used'
                    msg = msg.format(name)
                    log.warning(msg)
                    n += 1

    def _check_global_args(self, arg=None, task=None):
        args =  self.args if arg is None else [('', arg)]
        tasks =  self.tasks if task is None else [('', task)]
        value = True
        for name, arg in args:
            if name == 'task':
                continue
            for name, task in tasks:
                if arg.name not in task.args:
                    value = False
                    msg = 'warning: {}.args.{} not found in {}.args'
                    msg = msg.format(self.name, arg.name, task.name)
                    log.warning(msg)
        return value

    def _add_arg(self, name, unit='', default=None):
        arg = Argument(name, default, unit, task=self)
        self.args._append(name, arg)
        return arg

    def add_arg(self, name, unit='', default=None):
        arg = self._add_arg(name, unit, default)
        check = self._check_global_args(arg=arg)
        if check:
            for name, task in self.tasks:
                task.args[arg.name].depends_on(arg)

    @property
    def _task_current(self):
        return self.args.task.value

    @property
    def clen(self):
        return len(self.args.task._cache)

    def get_depend_args(self):
        return []

    @property
    def depend_tasks(self):
        return OrderedDict()

    def _configure(self):
        self.args._configure()
        self.args._nested._is_initialized = False
        self._task_current.args._configure()

    def _is_running(self):
        return (self.args._nested.is_running() or
                self._task_current.args._nested.is_running())

    def _is_finished(self):
        return not self._is_running()

    def next_value(self, loglevel=0, n=None):
        task = self._task_current
        if task.args._nested.is_finished():
            log.debug('{} was finished'.format(task))
            self.args._nested.next_value()
            for name, arg in self.args:
                arg._capture_idx()
            task = self._task_current
            task.args._configure()
            task.args._nested.reset()
        elif self.args._has_changed():
            for name, arg in self.args:
                arg._capture_idx()
        task.args._nested.next_value()
        if n == 0:
            self.call_log(loglevel)
        task._call()

    def call_log(self, loglevel=0):
        lines = []
        cols = []
        for name, arg in reversed(tuple(self.args)):
            if name == 'task':
                continue
            col = '{} = {}'.format(arg._uname, arg.value)
            if hasattr(arg._ptr, 'sweep'):
                sweep = arg._ptr.sweep
                col += ' ({}/{})'.format(sweep.idx + 1, len(sweep))
            cols.append(col)

        task = self._task_current
        msg = 'task'
        sweep = self.args.task._ptr.sweep
        if len(sweep) > 1:
            msg += ' {}/{}'.format(sweep.idx + 1, len(sweep))
        nested = task.args._nested
        msg += ':  {}'.format(task.name)
        if len(nested) > 1:
            msg += ' ({}/{})'.format(nested.idx + 1, len(nested))
        cols.append(msg)
        line = '    '.join(cols)
        lines.append(line)
        log.info('\n'.join(lines))

    def run(self, num=None, inner=False, plot_update=False):
        """run the sweeps of the taskmanager

        task.run()           run until task is finished

        task.run(0)          task.next_value()

        task.run(1)          next value of second sweep until
                             inner sweeps are finished

        task.run(1, True)    next value of second sweep until
                             the same inner sweep idx
        """
        if num is None and self._is_finished():
            self.reset()
        self._configure()
        plevel = plotmanager.log.level
        plotmanager.log.setLevel('WARNING')
        if plot_update and self._is_running():
            self.pm.open_window(0)
        inner_pre = [s.idx for s in self._get_sweeps(num)]
        loglevel = num if num is not None and num >= 0 else None
        n = 0
        while self._is_running():
            try:
                self.next_value(loglevel, n)
                if plot_update:
                    self._task_current.plot_update()
            except StopIteration:
                break
            if all([s.is_finished() for s in self._get_sweeps(num)]):
                    break
            n += 1
        if inner:
            while (self._is_running() and
                   ([s.idx for s in self._get_sweeps(num)] != inner_pre)):
                self.next_value(loglevel)
                if plot_update:
                    self._task_current.plot_update()
        if n > 0:
            if n > 1:
                log.info('...')
            self.call_log(loglevel)
        if not plot_update:
            if num is None or num < -1:
                for name, task in self.tasks:
                    task.plot_update()
            else:
                self._task_current.plot_update()
        plotmanager.log.level = plevel

    def _get_sweeps(self, num):
        task = self._task_current
        if num is None:
            sweeps = self.args._nested.sweeps[:num]
            sweeps.append(task.args._nested)
        elif num < 0:
            sweeps = self.args._nested.sweeps[:(-num - 1)]
            sweeps.append(task.args._nested)
        else:
            sweeps = task.args._nested.sweeps[:num]
        return sweeps

    def _to_dict(self):
        dct = OrderedDict()
        dct['__class__'] = self.__class__.__name__
        dct['name'] = self.name
        dct['tasks'] = [name for name, task in self.tasks]
        args = []
        for name, arg in self.args:
            if name == 'task':
                continue
            kwargs = OrderedDict()
            kwargs['name'] = arg.name
            kwargs['unit'] = arg._unit
            kwargs['default'] = arg._default.value
            args.append(kwargs)
        dct['args'] = args
        config = []
        for func, obj, args, kwargs in self._config:
            conf = OrderedDict()
            if isinstance(obj, TaskManager):
                conf['cmd'] = func.__name__
            else:
                conf['cmd'] = 'args.{}.{}'.format(obj.name, func.__name__)
            if args:
                conf['args'] = args
            if kwargs:
                conf['kwargs'] = kwargs
            config.append(conf)
        dct['config'] = config
        return dct

    def to_csv(self, dirname=''):
        if not dirname:
            dirname = self.name
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        filename = dirname + os.path.sep + self.name + '.csv'
        file = open(filename, 'w')
        for line in TabWriter(self).lines():
            file.write('## {}\n'.format(line))
        if self.clen:
            header = []
            header += ['arg_' + param._uname for _, param in self.args]
            header += ['state_' + name for name, param in self.args]
            file.write('; '.join(header) + '\n')
        for cidx in range(self.clen):
            line = []
            for name, param in self.args:
                line.append(repr(param.get_cache(cidx)))
            for name, param in self.args:
                line.append(repr(param._cache[cidx]))
            file.write('; '.join(line))
            file.write('\n')
        file.close()

        for name, task in self.tasks:
            filename = dirname + os.path.sep + name + '.csv'
            task.to_csv(filename)

    @classmethod
    def read_csv(cls, dirname='tm'):
        file = open(dirname + os.path.sep + dirname + '.csv')
        jlines = []
        for line in file:
            if line.startswith('##'):
                jlines.append(line.strip('#'))
            else:
                break
        dct = json.loads('\n'.join(jlines))

        # create taskmanager
        tm = cls(name=dct['name'])
        # create global tm.args
        for kwargs in dct['args']:
            tm.add_arg(**kwargs)
        # apply config of tm
        for conf in dct['config']:
            names = conf['cmd'].split('.')
            if len(names) == 1:
                func_name = names[0]
                func = getattr(tm, func_name)
            else:
                param_namespace, param_name, func_name = names
                params = getattr(tm, param_namespace)
                param = getattr(params, param_name)
                func = getattr(param, func_name)
            _args = conf.get('args', ())
            _kwargs = conf.get('kwargs', {})
            func(*_args, **_kwargs)

        # read arg state from csv
        args = [arg for name, arg in tm.args]
        nargs = len(args)
        for cidx, line in enumerate(file):
            columns = line.strip('\n').split('; ')
            acolumns = columns[nargs:]
            for arg, col in zip(args, acolumns):
                value = ast.literal_eval(col)
                arg._cache.append(value)
                arg._states.setdefault(value, set()).add(cidx)
        file.close()

        namespace = {tm.name: tm}
        tasks = {}  # name: dct, task
        for taskname in dct['tasks']:
            filename = dirname + os.path.sep + taskname + '.csv'
            dct, task = Task._read_csv(filename)
            tasks[taskname] = dct, task
            tm._append(task)
            namespace[taskname] = task
        for name, (dct, task) in tasks.items():
            task._apply_config(dct, namespace)

        # arg state recover
        tm.args._configure()
        tm.args._nested._is_initialized = False
        for name, arg in tm.args:
            if arg._cache:
                state = arg._cache[-1]
                arg.state = state
        return tm


class AxesManager(object):
    def __init__(self):
        self.rows = 1
        self.cols = 1
        self.pms = {}
        self.axes = {}
        self.lines = {}
        self.fig = None

    @staticmethod
    def _loc(loc):
        if isinstance(loc, (tuple, list)):
            if loc[1] is not None:
                return (loc[0], loc[1]+1)
            else:
                return (loc[0], loc[1])
        elif loc is None:
            return (None, None)
        else:
            return (loc, loc+1)

    def append_plot_manager(self, pm, row, col):
        self.pms[pm] = row, col
        ridx = self._loc(row)
        cidx = self._loc(col)
        self.rows = max(self.rows, 1 if ridx[1] is None else ridx[1])
        self.cols = max(self.cols, 1 if cidx[1] is None else cidx[1])

    def get_axes(self, pm, xax=None):
        if self.fig is None:
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('pick_event', self.on_pick)
            #~ self.add_subplot_zoom(self.fig)
        try:
            return self.axes[pm]
        except KeyError:
            row, col = self.pms[pm]
            ridx = self._loc(row)
            cidx = self._loc(col)
            gs = GridSpec(self.rows, self.cols)
            ax = self.fig.add_subplot(gs[slice(*ridx), slice(*cidx)],
                                      sharex=xax)
            self.axes[pm] = ax
            return ax

    def update(self):
        for ax in self.fig.axes:
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def on_pick(self, event):
        line = event.artist
        idx = event.ind[0]
        pm = self.lines[line]
        xidx, yidx = pm.lines[line][idx]
        for pm in self.pms:
            xval = pm.xarg[xidx]
            yval = pm.yarg[yidx]
            pm.set_cursor(xval, yval, line.get_color())
        self.update()


class BasePlotManager:
    def __init__(self):
        pass

class SweepPlotManager(BasePlotManager):
    def __init__(self, xarg, yarg, use_cursor=True,
                 xlim=(None, None), xlabel='', ylabel='',
                 sorted_xdata=False, am=None, **kwargs):
        self.xarg = xarg
        self.yarg = yarg
        self.xlabel = xlabel if xlabel else xarg.name
        self.ylabel = ylabel if ylabel else yarg.name
        self.xlim = xlim
        self.use_cursor = use_cursor
        self.sorted_xdata = sorted_xdata
        self.am = am
        self.kwargs = kwargs
        self.reset()

        super().__init__()

    def reset(self):
        self.line = None
        self.lines = {}
        #~ self.skeys = {}
        #~ self.keys = {}
        self.cursor = None
        self.annotation = None

    def __repr__(self):
        args = []
        for name in 'xarg', 'yarg', 'xlabel', 'ylabel', 'xlim':
            arg = '{}={}'.format(name, repr(getattr(self, name)))
            args.append(arg)
        if not self.use_cursor:
            arg = 'use_cursor=False'
            args.append(arg)
        if self.sorted_xdata:
            arg = 'sorted_xdata=True'
            args.append(arg)
        for name, value in self.kwargs.items():
            arg = '{}={}'.format(name, repr(value))
            args.append(arg)
        return "{classname}({args})".format(
            classname=self.__class__.__name__,
            args=', '.join(args))

    def _max(self, val):
        if val is None:
            return -1

    def _pop(self, idxs, items):
        idx = idxs[0]
        ipos = idx if idx >= 0 else len(items) + idx
        if len(idxs) > 1:
            return items[:ipos] + (self._pop(idxs[1:], items[ipos]),) + items[ipos +1:]
        else:
            return items[:ipos] + items[ipos+1:]

    @property
    def ax(self):
        return self.am.get_axes(self)

    def plot(self):
        self.lines[self.line].append((self.xarg.idx, self.yarg.idx))

        xval = self.xarg.value
        yval = self.yarg.value
        yval = np.nan if yval is None else yval

        xdata = self.line.get_xdata()
        xdata = np.append(xdata, xval)
        self.line.set_xdata(xdata)

        ydata = self.line.get_ydata()
        ydata = np.append(ydata, yval)
        self.line.set_ydata(ydata)

        self.set_cursor(xval, yval, self.line.get_color())
        self.am.update()

    def newline(self):
        kwargs = dict(self.kwargs)
        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'
        line, = self.ax.plot([], [], picker=10, **kwargs)
        self.line = line
        self.am.lines[line] = self
        self.lines[line] = []
        return line

    def set_cursor(self, xval, yval, color):
        if not self.use_cursor:
            return
        if self.cursor is None:
            self.cursor, = self.ax.plot(np.nan, np.nan,
                marker='o',
                markersize=17.5,
                alpha=0.5)
        self.cursor.set_data([xval, yval])
        #~ self.cursor.set_color(shade_color(line.get_color(), 50))
        self.cursor.set_color(color)
        if self.annotation is None:
            self.annotation = self.ax.annotate(
                s='anno',
                xy=(np.nan, np.nan),
                xytext=(0, -15),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=11 ,
                bbox=dict(
                    boxstyle='round,pad=0.25',
                    alpha=0.9,
                    edgecolor='none',
                ),
                visible=False,
            )
        self.annotation.xy = xval, yval
        bbox = self.annotation.get_bbox_patch()
        bbox.set_facecolor(self._shade_color(color, 50))
        self.annotation.set_visible(False)

    def update_cursor(self, key, args, output, **kwargs):
        line, idx = self.keys[key]
        x = line.get_xdata()[idx]
        y = line.get_ydata()[idx]
        self.set_cursor(x, y, line.get_color())
        return y

    def draw(self):
        ax = self.ax
        if self.cursor is not None:
            ax.add_line(self.cursor)
        if self.annotation is not None:
            ax._add_text(self.annotation)
        for line in self.lines:
            ax.add_line(line)

    @staticmethod
    def _shade_color(color, percent):
        """ A color helper utility to either darken or lighten given color

        from https://github.com/matplotlib/matplotlib/pull/2745
        """
        rgb = colorConverter.to_rgb(color)
        h, l, s = rgb_to_hls(*rgb)
        l *= 1 + float(percent)/100
        l = np.clip(l, 0, 1)
        r, g, b = hls_to_rgb(h, l, s)
        return r, g, b


if __name__ == '__main__':
    import doctest
    doctest.testmod(
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)

    #~ import test_basics
    #~ test_basics.test_select_by_sweep()

    utils.enable_logger(plotmanager.log)
    utils.enable_logger(log, 'info', 'short')
    #~ utils.enable_logger(log, 'debug', 'short')


    def func(x, temp=25):
        return

    def sigma(temp=25):
        value = 0.1
        return value

    def offs(x, sigma=0.1, temp=25):
        y = x * np.random.normal(1, sigma) + temp
        return y

    def quad(x, m=1, n=0, a=0, b=1, c: 'cm' = 2, temp=25):
        """my quad function"""
        y = m * x**2 + a*x + n + temp
        z: 'cm' = 4 + 5j
        return y, z

    tm = TaskManager()
    tm.add_arg('temp', unit='deg')
    tm.args.temp.iterate(0, 100, 200)

    func = tm.append(func)
    func.args.x.sweep(1, 3)

    sigma = tm.append(sigma)

    offs = tm.append(offs)
    offs.args.x.sweep(0, 5)
    offs.args.sigma.depends_on(sigma.returns.value)
    offs.plot('x', 'y')

    quad = tm.append(quad)
    quad.args.x.sweep(0, 12)
    quad.args.m.iterate(5, 10)
    quad.args.n.depends_on(offs.returns.y)
    #~ quad.args.n.depends_on(offs.returns.y, sweeps='x')
    #~ quad.args.c.depends_on(offs.returns.y, squeeze='x')
    #~ quad.args.a.sweep(10, 30, 10)
    #~ quad.args.a.zip('m')

    quad.plot('x', 'y')

    tm.run(-2)
    tm.run(-2)
    tm.run(-1)
    tm.run(-1)
    tm.run(0)
    tm.run(0)
    tm.run(0)

    tm.to_csv()
    tm2 = TaskManager.read_csv()


if 0:
    #~ quad.to_csv()
    @Task
    def mytask(x=1, y=2, offs=0):
        z = x + y + offs
        return z

    mytask.args.x.sweep(0, 4)
    mytask.args.y.sweep(10, 30, step=5)
    mytask.args.offs.value = 1234
    mytask.args.offs.iterate(0, 100)
    mytask.args.x.zip('y')
    mytask.run(1)
    #~ mytask.run(1)
    #~ mytask.run(1)
    mytask.run(0)
    mytask.run(0)

    mytask.to_csv()
    mt = Task.read_csv('mytask.csv')


