import loopyplot
from loopyplot import Task
from loopyplot.taskmanager import BaseSweepIterator


class TaskSweep(BaseSweepIterator):
    def __init__(self, task):
        self.task = task
        self.clen = 0
        self.last_clen = 0
        super().__init__()

    def get_value(self, idx):
        cidx = self.last_clen + idx
        return cidx

    def __len__(self):
        return self.clen - self.last_clen

    def configure(self):
        self.last_clen = self.clen
        self.clen = self.task.clen
        self.idx = 0


class DependParamPointer:
    def __init__(self, param, tasksweep):
        msg = 'param and tasksweep must have the same task'
        assert param._task is tasksweep.task, msg
        self._param = param
        self._tasksweep = tasksweep

    @property
    def state(self):
        cidx = self._tasksweep.value
        arg_state = self._param._cache[cidx]
        return arg_state

    def get_value(self, state):
        return self._param.get_value(state)

    @property
    def value(self):
        return self.get_value(self.state)

"""

class ReturnParam:



    @property
    def state(self):
        return len(self._cache)

    def get_state(self, cidx):
        clen = cidx + 1
        return clen

    def get_value(self, state):
        clen = state
        if clen > 0:
            return self._cache[clen - 1]
        else:
            return np.nan

    #### old ####


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


"""


if __name__ == '__main__':

    def poly(x, a=0, b=0):
        y = (x - a)*(x - b)
        return y

    poly = Task(poly)
    poly.args.x.sweep(0, 2, num=7)
    poly.args.a.iterate(0.25, 0.75)
    poly.args.b.iterate(1.25, 1.75)
    #~ poly.args.b.iterate(0.5, 2)
    #~ poly.args.c.iterate(-1, 2)

    #~ poly.plot('x', 'y')



    t = TaskSweep(poly)
    p = DependParamPointer(poly.args.x, t)

    poly.run(1)

    t.configure()
    for idx in t:
        print(p.value)
