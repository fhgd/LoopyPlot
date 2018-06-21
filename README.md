LoopyPlot
=========

Plot data dependencies for scientific and engineering tasks in Python. With ...

> Loo **_py_** Plot ... you can describe your problem in a plain
> [python](https://www.python.org) function
>
> **_Loop_** yPlot ... you can run nested parameter sweeps with data
> dependencies
>
> Loopy **_Plot_** ... you can explore the results by interactive plots
> (many thanks to [matplotlib](https://matplotlib.org/))
>
> **_Loopy_** Plot ... you just configure what you want and everything
> else is done in the background for you


Motivation
----------

In the field of engineering science one common experiment task is to
explore the behavior of a black-box under different inputs. The
black-box can be a pure mathematical function, a numerical algorithm,
the results of a complex simulation or even an experimental measurement.
In many cases the input variation are done by nested for-loops.

While a nested for-loop iteration is simple to code the data management
can be become quite complicated. This is even true when you want

1.  to quickly change the *loop configuration* (nested loops vs. zipped
  loops)
2.  to define *data dependencies* between different experiments
3.  to have an *error recovery* of the loop state because each
  iteration step takes a reasonable amount of time
4.  to *plot* multiple curves preserving the *relations* among each other
5.  to have a *live-update* of all plots while running the loop iteration
6.  to *save* experiment data in a readable and consistent way
7.  to *reload* the interactive plots later from an already finished
   experiment
8.  to write *readable code* which can be shared for collaboration

Especially the last point requires you to split the specific part of
an experiment from its administration (all the seven point above). A very
natural way of splitting is to use a function. Everything inside the
function describes the specific experiment. The function arguments and
return values are used for the administration of the experiment.
In this sense LoopyPlot is a dependency injection (DI) container for
looped functions with plotting evaluation of the results.

LoopyPlot is a prototype implementation of the seven administration
points above in order the write readable experiments.


Short Demo
-----------

The following basic steps are shown by means of the nice lissajous example:

* Configure the function arguments either with constant values or loops
* Run the nested loops
* Plot the function return values and show the data cursor

First of all we need to write the experiment as a python function.
Here we use two sinusoidal functions in order to create the lissajous
curves.

```python
from numpy import pi, cos, sin

def lissajous(t, freq=2, phi=0):
    xpos = sin(2*pi * t)
    ypos = cos(2*pi*freq * t + phi)
    return xpos, ypos
```

In order to explore the behavior of your lissajous function we
sweep the argument `t` and `phi`.

```python
from loopyplot import TaskManager

# append the lissajous function as a task to the taskmanager
tm = TaskManager()
task = tm.append(lissajous)

# (1) configure the parameter sweeps
task.args.t.sweep(0, 1, num=30)
task.args.phi.iterate(pi/4, pi/2, 3*pi/4)
```

Afterwards we can run the double sweep and see the results.


```python
# (2) run the the configured lissajous task
tm.run()

# (3) display the results
task.plot('t', 'xpos', row=0, col=1)
task.plot('t', 'ypos', row=1, col=1)
task.plot('xpos', 'ypos', squeeze='t', accumulate='t', row=None)
```

The matplotlib figure has an interactive data cursor.

![Lissajous](./examples/lissajous.gif)

You can click at any point (e.g. in the lower right axes) in order to
update the data cursor and explore the relations between the plots.


Install
-------

In order instal LoopyPlot, simply download the repository, change into
the folder `LoopyPlot` and

    pip install . --user

If necessary this will automatically install the dependencies:

* matplotlib
* pandas
* numpy

LoopyPlot is developed under python 3.6 (older versions are not tested)
and intended to use with an ipython shell (5.7.0):

    ipython --pylab


Features
--------

TaskManager:
* convert a pure python function into a task by extracting the functions
  arguments (with default values) and return values
* each parameter could have a unit annotation
  (new variable annotation in python 3.6)
* manage multiple tasks in a list
* run the whole task list completely or step-wise
* save the task configuration and the results in csv files
* reload the task from csv file (used for plotting)
* experimental: global arguments of a task list

Configuration of function arguments:
* arguments can be either:
    - a constant value,
    - a linear sweep,
    - a sequence of user defined values or
    - a (looped) parameter dependency
* by default argument loops are nested in the order of definition
* multiple loops can be zipped togehter in one common loop
* loops can be concatenated

PlotManager:
* live update during a running task loop
* data cursor which highlights the relations between different plots
* manage multiple plot-views
* each view can be shown in one or multiple windows



Roadmap
-------

Open issues before first beta release 0.1:
* write more use case demos
* docstings

PlotManager:
* connect common axes for jointly zooming
* sweeps: add new data points in arbitrary order (not only append)

TaskManager:
* csv: split arrays in extra csv file (instead of string format)

Further ideas (beyond 0.1):
* implement horizontal nested task lists over vertical looped lists

