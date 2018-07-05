import loopyplot
from loopyplot import Task
from loopyplot.taskmanager import BaseSweepIterator

import random


@Task
def poly(x, a=0, b=0):
    y = (x - a)*(x - b)
    return y

poly.args.x.sweep(0, 2, num=7)
poly.args.a.iterate(0.25, 0.75)
poly.args.b.iterate(1.25, 1.75)
#~ poly.args.b.iterate(0.5, 2)
#~ poly.args.c.iterate(-1, 2)

#~ poly.plot('x', 'y')

#~ poly.run(1)
#~ df = poly.returns.as_table()
#~ print(df)

@Task
def noise(x, sigma=0.2, a=0):
    y = random.gauss(x, sigma)
    return y

#~ noise.args.x.sweep(0, 5, step=poly.returns.y)
noise.args.add_depending_task(poly)
noise.args.x.depends_on_param(poly.returns.y)
noise.args.a.depends_on_param(poly.args.a)
noise.args.sigma.iterate(0.2, 1)

#~ noise.run(2)
#~ poly.run(1)
#~ noise.run(2)

#~ poly.run()
#~ noise.run()

poly.run(1)
noise.run(1)
