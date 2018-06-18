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
black-box can be a pure mathematical function, a numeric algorithm,
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

LoopyPlot is a prototype implementation of the seven administration
points above in order the write readable experiments.


Simple Demo
-----------

This demo only shows the following steps:

* configure the function arguments either with constant values or loops
* configure the plot arrangement by using the function return values
* run all the experiment loops either step-by-step or completely

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

# configure the parameter sweeps
task.args.t.sweep(0, 1, num=30)
task.args.phi.iterate(pi/4, pi/2, 3*pi/4)
```

Afterwards we can run the double sweep and see the results.


```python
# run the the configured lissajous task
tm.run()

# display the results
task.plot('t', 'xpos', row=0, col=1)
task.plot('t', 'ypos', row=1, col=1)
task.plot('xpos', 'ypos', squeeze='t', accumulate='t', row=None)
```

The matplotlib figure has an interactive data cursor.

![Lissajous](./examples/lissajous.gif)

You can click at any point (in the lower right axes) in order to
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
