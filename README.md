# LoopyPlot
Plot nested loop data for scientific and engineering tasks in Python.

LoopyPlot allows you to concentrate on writing your task in a plain
python function like

    >>> def quad(x, offs=0):
    ...     y = x**2 + offs
    ...     return y

In order to explore the behavior of your sophisticated function by
plotting diffent sweeps you can write

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

Without LoopyPlot you have to write the usual spaghetti-code like

    >>> import matplotlib.pylab as plt
    >>> import numpy as np

    >>> data = {}
    ... for _offs in offs:
    ...     for _x in x:
    ...         y = data.setdefault(_offs, [])
    ...         value = quad(_x, _offs)
    ...         y.append(value)
    >>> data
    {0: [0.0, 6.25, 25.0, 56.25, 100.0],
     60: [60.0, 66.25, 85.0, 116.25, 160.0]}

    >>> for _offs, y in data.items():
    ...     plt.plot(x, y, label='offs = {}'.format(_offs))
    >>> plt.legend()

in order to explore your function graphically.
