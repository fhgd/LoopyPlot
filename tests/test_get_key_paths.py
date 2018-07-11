import pytest


def test_get_key_paths_1(task_setup):
    one, two, three = task_setup

    """
        +---------------------+
        |  three: x1, x2, x3  |
        +---------------------+
          |         |
          |         | squeeze:
          |         |   (2.)     [two.args.x]
          |         |   (3.) or  [two, one.args.x]
          |         |   (4.) or [[two, one.args.offs], [two.args.offs]]
          |         V
          |      +----------------+
          |      |  two: x, offs  |
          |      +----------------+
          |                  |
          | squeeze: 'offs'  | squeeze: 'x'
          |   (1.)           |   (1.)
          V                  V
        +----------------------+
        |  one: x, gain, offs  |
        +----------------------+
    """


    tasksweep = two.args._tasksweeps[one]
    assert tasksweep.squeeze == [[one.args.x]]
    key_paths = tasksweep.get_key_paths()
    assert key_paths == [[one.args.gain], [one.args.offs]]

    tasksweep = three.args._tasksweeps[one]
    assert tasksweep.squeeze == [[one.args.offs]]
    key_paths = tasksweep.get_key_paths()
    assert key_paths == [[one.args.x], [one.args.gain]]


def test_get_key_paths_2(task_setup):
    one, two, three = task_setup

    three.add_dependency(two, squeeze=[[two.args.x]])
    three.args.x1.depends_on_param(two.args.offs)

    tasksweep = three.args._tasksweeps[two]
    assert tasksweep.squeeze == [[two.args.x]]
    key_paths = tasksweep.get_key_paths()
    assert key_paths == [[two.args.offs]]


def test_get_key_paths_3(task_setup):
    one, two, three = task_setup

    three.add_dependency(two, squeeze=[[two.args.x, one.args.x]])
    three.args.x1.depends_on_param(two.args.offs)

    tasksweep = three.args._tasksweeps[two]
    assert tasksweep.squeeze == [[two.args.x, one.args.x]]
    key_paths = tasksweep.get_key_paths()
    assert key_paths == [
        [two.args.x, one.args.gain],
        [two.args.x, one.args.offs],
        [two.args.offs]]


def test_get_key_paths_4(task_setup):
    one, two, three = task_setup

    three.add_dependency(two, squeeze=[
        [two.args.x, one.args.offs],
        [two.args.offs],
    ])
    three.args.x1.depends_on_param(two.args.offs)

    tasksweep = three.args._tasksweeps[two]
    assert tasksweep.squeeze == [[two.args.x, one.args.offs], [two.args.offs]]
    key_paths = tasksweep.get_key_paths()
    assert key_paths == [[two.args.x, one.args.gain]]


@pytest.fixture
def task_setup():
    from loopyplot import Task
    import random

    @Task
    def one(x, gain=1, offs=1):
        y = gain*x + offs
        return y
    one.args.x.sweep(0, 2)
    #~ one.args.gain.iterate(1, 3)
    #~ one.args.offs.iterate(0, 5)

    @Task
    def two(x, offs=0):
        y = x**2 + offs
        return y
    two.add_dependency(one, squeeze=[[one.args.x]])
    two.args.x.depends_on_param(one.returns.y)
    #~ two.args.a.depends_on_param(one.args.offs)
    two.args.offs.iterate(0, 10)

    @Task
    def three(x1, x2=[1, 2, 3], x3=3):
        y = x1 * sum(x2) + x3
        return y
    three.add_dependency(one, squeeze=[[one.args.offs]])
    three.args.x2.depends_on_param(one.returns.y)

    return one, two, three
