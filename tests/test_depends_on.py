import pytest


@pytest.fixture
def task_setup():
    from loopyplot import Task

    @Task
    def quad(x, offs=0):
        y = x*x + offs
        return y
    quad.args.x.sweep(1, 4)
    quad.args.offs.iterate(0, 100)
    quad.run()

    @Task
    def two(a, b):
        return

    return quad, two


def test_without_squeeze(task_setup):
    quad, two = task_setup

    df = quad.returns.as_table()
    lines = ['   x  offs    y',
             '0  1     0    1',
             '1  2     0    4',
             '2  3     0    9',
             '3  4     0   16',
             '4  1   100  101',
             '5  2   100  104',
             '6  3   100  109',
             '7  4   100  116']
    assert repr(df).split('\n') == lines

    two.add_dependency(quad)
    two.args.b.depends_on(quad.returns.y)

    two.run()
    df = two.returns.as_table(hide_const=True)
    lines = ['     b',
             '0    1',
             '1    4',
             '2    9',
             '3   16',
             '4  101',
             '5  104',
             '6  109',
             '7  116']
    assert repr(df).split('\n') == lines


def test_squeeze(task_setup):
    quad, two = task_setup

    df = quad.returns.as_table()
    lines = ['   x  offs    y',
             '0  1     0    1',
             '1  2     0    4',
             '2  3     0    9',
             '3  4     0   16',
             '4  1   100  101',
             '5  2   100  104',
             '6  3   100  109',
             '7  4   100  116']
    assert repr(df).split('\n') == lines

    two.args.a.iterate(2, 4)
    two.add_dependency(quad, squeeze=quad.args.x)
    two.args.b.depends_on(quad.returns.y)

    two.run()
    df = two.returns.as_table()
    lines = ['   a                     b',
             '0  2         [1, 4, 9, 16]',
             '1  4         [1, 4, 9, 16]',
             '2  2  [101, 104, 109, 116]',
             '3  4  [101, 104, 109, 116]']
    assert repr(df).split('\n') == lines


def test_double_squeeze(task_setup):
    quad, two = task_setup

    df = quad.returns.as_table()
    lines = ['   x  offs    y',
             '0  1     0    1',
             '1  2     0    4',
             '2  3     0    9',
             '3  4     0   16',
             '4  1   100  101',
             '5  2   100  104',
             '6  3   100  109',
             '7  4   100  116']
    assert repr(df).split('\n') == lines

    two.args.a.iterate(2, 4)
    two.add_dependency(quad, squeeze=[quad.args.x, quad.args.offs])
    two.args.b.depends_on(quad.returns.y)

    sq_path = two.args._last_tasksweep.squeeze
    assert sq_path == [[quad.args.x], [quad.args.offs]]

    two.run()
    df = two.returns.as_table()
    lines = ['   a                                  b',
             '0  2  [1, 4, 9, 16, 101, 104, 109, 116]',
             '1  4  [1, 4, 9, 16, 101, 104, 109, 116]']
    assert repr(df).split('\n') == lines

