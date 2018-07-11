from loopyplot import Task
import random


@Task
def zeros(mid=1, gap=0.5):
    x01 = mid - gap
    x02 = mid + gap
    return x01, x02

zeros.args.mid.iterate(0.75, 1.25)


@Task
def poly(x, x01=0, x02=0):
    y = (x - x01)*(x - x02)
    return y

poly.args.x.sweep(0, 2, num=25)
poly.add_dependency(zeros)
poly.args.x01.depends_on(zeros.returns.x01)
poly.args.x02.depends_on(zeros.returns.x02)



@Task
def noise(x: 'cm', sigma=0.2, s=0.1):
    y: 'cm' = random.gauss(x, sigma*s)
    return y

noise.add_dependency(poly)
noise.args.x.depends_on(poly.returns.y)

noise.add_dependency(zeros)
noise.args.sigma.depends_on(zeros.returns.x01)

#~ noise.args.sigma.iterate(0.01, 0.05)
#~ noise.args.sigma.iterate(0.01, 0.05)
#~ noise.args.s.iterate(0.1, 0.2)
#~ noise.args.zip('sigma', 's')

if 1:
    zeros.run()
    poly.run()
    noise.run()

#~ poly.plot('x', 'y')


if 0:
    noise.plot(poly.args.x, 'y')
    #~ noise.plot(poly.args.x, 'x', accumulate='')
    noise.plot(poly.args.x, 'x', accumulate=[[zeros.args.mid, poly]])
    #~ lm = noise.pm.lms[0, 0, 0][noise][1]
