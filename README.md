LoopyPlot
=========

Plot nested loop data for scientific and engineering tasks in Python.

ToDo: motivation

Usage
=====

LoopyPlot allows you to concentrate on writing your task in a plain
python function like this

```python
from numpy import pi, cos, sin

def lissajous(t, freq=2, phi=0):
    xpos = sin(2*pi * t)
    ypos = cos(2*pi*freq * t + phi)
    return xpos, ypos
```

In order to explore the behavior of your sophisticated function by
plotting diffent sweeps you can write

```python
from loopyplot import TaskManager

# append the lissajous function as a task to the taskmanager
tm = TaskManager()
task = tm.append(lissajous)

# configure the inner and outer parameter sweeps
task.args.t.sweep(0, 1, num=30)
task.args.phi.iterate(pi/4, pi/2, 3*pi/4)

# run the sweeps of the configured lissajous task
tm.run()

# display the results
task.plot('t', 'xpos', row=0, col=1)
task.plot('t', 'ypos', row=1, col=1)
task.plot('xpos', 'ypos', squeeze='t', accumulate='t', row=None)
```

which results in a matplotlib figure with an interactive data cursor

![Lissajous](./examples/lissajous.gif)

You can click at any data point (in the lower right axes) in order to
update the data cursor and explore the relations in your plotted data.


Install
=======

In order instal LoopyPlot, simply download the repository, change into
the folder `LoopyPlot` and

    pip install . --user

If necessary this will automatically install the dependencies:

* numpy
* pandas
* matplotlib
