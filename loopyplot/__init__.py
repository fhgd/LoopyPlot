"""LoopyPlot -- Plot nested loops for scientific and engineering tasks

How to use LoopyPlot
====================

Basic usage

    >>> import loopyplot as lp
    >>> def quad(x, offs=0):
    ...     y = x**2 + offs
    ...     return y
    ...
    >>> tm = lp.TaskManager(quad)
    >>> tm.tasks.quad.args.x.sweep(0, 10, num=5)
    >>> tm.tasks.quad.args.offs.iterate(0, 60)
    >>> tm.run()
    >>> tm.tasks.quad.returns.as_table()
          x  offs       y
    0   0.0     0    0.00
    1   2.5     0    6.25
    2   5.0     0   25.00
    3   7.5     0   56.25
    4  10.0     0  100.00
    5   0.0    60   60.00
    6   2.5    60   66.25
    7   5.0    60   85.00
    8   7.5    60  116.25
    9  10.0    60  160.00


And now plot the data with

    >>> tm.tasks.quad.plot('x', 'y')

which results in a matplotlib figure with an interactive data cursor.
"""


from .taskmanager import Task, TaskManager, log
from .plotmanager import PlotManager
from .plotmanager import log as log_pm

from .utils import enable_logger
enable_logger(log, 'info', 'short')
enable_logger(log_pm, 'info', 'short')
