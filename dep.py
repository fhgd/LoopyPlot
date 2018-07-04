import loopyplot
from loopyplot import Task
from loopyplot.taskmanager import BaseSweepIterator

import random


@Task
def zeros(mid=1, d=0.5):
    a = mid - d
    b = mid + d
    return a, b

zeros.args.mid.iterate(0.75, 1.25)

@Task
def poly(x, a=0, b=0):
    y = (x - a)*(x - b)
    return y

poly.args.x.sweep(0, 2, num=25)
poly.args.add_depending_task(zeros)
poly.args.a.depends_on_param(zeros.returns.a)
poly.args.b.depends_on_param(zeros.returns.b)
#~ poly.args.b.iterate(1.25)
#~ poly.args.b.iterate(1.25, 1.75)
#~ poly.args.b.iterate(0.5, 2)
#~ poly.args.c.iterate(-1, 2)

#~ poly.plot('x', 'y')

#~ poly.run(1)
#~ df = poly.returns.as_table()
#~ print(df)

@Task
def noise(x, sigma=0.2, s=0.2, a=0):
    y = random.gauss(x, sigma)
    return y

#~ noise.args.x.sweep(0, 5, step=poly.returns.y)
noise.args.add_depending_task(poly)
noise.args.x.depends_on_param(poly.returns.y)
noise.args.a.depends_on_param(poly.args.a)
#~ noise.args.sigma.iterate(0.1)
noise.args.sigma.iterate(0.01, 0.05)
noise.args.s.iterate(1, 5)
noise.args.zip('sigma', 's')

#~ noise.plot(poly.args.x, 'x')
noise.plot(poly.args.x, 'y')

zeros.run()
poly.run()
noise.run()


#~ noise.run(2)
#~ poly.run(1)
#~ noise.run(2)

#~ poly.run(1)
#~ noise.run(1)
