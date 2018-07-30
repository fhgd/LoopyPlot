import pytest


""" create test plan by means of the dependency graph:

    +------------------------+
    |  four: (x1, x2) --> y  |
    +------------------------+
       (x1, x2):|
                |  squeeze:
                |    (6.)  []
                |
                |
                |:(x1, x2)
                V
    +-------------------------+
    |  three: (x1, x2) --> y  |
    +-------------------------+
     x1:|    x2:|
        |       | squeeze:
        |       |   (1.)  []
        |       |   (2.)  [two.args.offs]
        |       |   (3.)  [two.args.x]
        |       |   (4.)  [two.args.x, one.args.offs]
        |       |   (5.)  [two.args.x, one.args.offs], [two.args.offs]
        |       |   (6.)  [two.args.x, one.args.offs], [two.args.offs]
        |       |
        |       |:y
        |       V
        |    +------------------------+
        |    |  two: (x, offs) --> y  |
        |    +------------------------+
        |                 x:|
        | squeeze:          | squeeze:
        |   [one.args.x]    |   [one.args.x]
        |                   |
        |:y                 |:y
        V                   V
    +------------------------------+
    |  one: (x, gain, offs) --> y  |
    +------------------------------+
"""


@pytest.fixture
def task_setup():
    from loopyplot import Task

    @Task
    def one(x, gain=1, offs=1):
        y = gain*x + offs
        return y
    one.args.x.sweep(0, 2)

    @Task
    def two(x, offs=0):
        y = sum(x**2 + offs)
        return y
    two.args.offs.iterate(0, 10)
    two.add_dependency(one, squeeze=one.args.x)
    two.args.x.depends_on(one.returns.y)

    @Task
    def three(x1, x2):
        y = sum(x1) * x2
        return y
    three.add_dependency(one, squeeze=one.args.x)
    three.args.x1.depends_on(one.returns.y)

    @Task
    def four(x1, x2):
        y = sum(x1) + sum(x2)
        return y

    return one, two, three, four


def test_1(task_setup):
    one, two, three, four = task_setup

    one.run()
    df = one.returns.as_table(hide_const=True)
    lines = ['   x  y',
             '0  0  1',
             '1  1  2',
             '2  2  3']
    assert repr(df).split('\n') == lines

    two.run()
    df = two.returns.as_table()
    lines = ['           x  offs   y',
             '0  [1, 2, 3]     0  14',
             '1  [1, 2, 3]    10  44']
    assert repr(df).split('\n') == lines

    three.add_dependency(two)
    three.args.x2.depends_on(two.returns.y)

    # this is the same like
    #
    #     three.add_dependency(two, squeeze=[
    #         [two.args.x, one.args.x]
    #     ])
    #
    # because one.args.x is already squeeze by two and
    # is consequently not avialable in key_paths:

    key_paths = three.args.x2._ptr._tasksweep.get_arg_paths()
    assert key_paths == [[two.args.x, one.args.gain],
                         [two.args.x, one.args.offs],
                         [two.args.offs]]

    # currently no support for squeezing the same task in different ways
    #~ three.add_dependency(one)
    #~ three.args.x3.depends_on(one.returns.y)

    three.run()
    df = three.returns.as_table()
    lines = ['          x1  x2    y',
             '0  [1, 2, 3]  14   84',
             '1  [1, 2, 3]  44  264']
    assert repr(df).split('\n') == lines


def test_2(task_setup):
    one, two, three, four = task_setup

    one.run()
    df = one.returns.as_table(hide_const=True)
    lines = ['   x  y',
             '0  0  1',
             '1  1  2',
             '2  2  3']
    assert repr(df).split('\n') == lines

    two.run()
    df = two.returns.as_table()
    lines = ['           x  offs   y',
             '0  [1, 2, 3]     0  14',
             '1  [1, 2, 3]    10  44']
    assert repr(df).split('\n') == lines

    three.add_dependency(two, squeeze=two.args.offs)
    three.args.x2.depends_on(two.returns.y)

    three.run()
    df = three.returns.as_table()
    lines = ['          x1        x2          y',
             '0  [1, 2, 3]  [14, 44]  [84, 264]']
    assert repr(df).split('\n') == lines


def test_3(task_setup):
    one, two, three, four = task_setup

    one.args.offs.iterate(1, 10)
    one.run()
    df = one.returns.as_table(hide_const=True)
    lines = ['   x  offs   y',
             '0  0     1   1',
             '1  1     1   2',
             '2  2     1   3',
             '3  0    10  10',
             '4  1    10  11',
             '5  2    10  12']
    assert repr(df).split('\n') == lines

    two.run()
    df = two.returns.as_table()
    lines = ['              x  offs    y',
             '0     [1, 2, 3]     0   14',
             '1     [1, 2, 3]    10   44',
             '2  [10, 11, 12]     0  365',
             '3  [10, 11, 12]    10  395']
    assert repr(df).split('\n') == lines

    three.add_dependency(two, squeeze=two.args.x)
    three.args.x2.depends_on(two.returns.y)

    three.run()
    df = three.returns.as_table()
    lines = ['             x1         x2              y',
             '0     [1, 2, 3]  [14, 365]     [84, 2190]',
             '1  [10, 11, 12]  [14, 365]   [462, 12045]',
             '2     [1, 2, 3]  [44, 395]    [264, 2370]',
             '3  [10, 11, 12]  [44, 395]  [1452, 13035]']
    assert repr(df).split('\n') == lines


def test_4(task_setup):
    one, two, three, four = task_setup

    one.args.gain.iterate(1, 10)
    one.args.offs.iterate(0, 25)
    one.run()
    df = one.returns.as_table()
    lines = ['    x  gain  offs   y',
             '0   0     1     0   0',
             '1   1     1     0   1',
             '2   2     1     0   2',
             '3   0    10     0   0',
             '4   1    10     0  10',
             '5   2    10     0  20',
             '6   0     1    25  25',
             '7   1     1    25  26',
             '8   2     1    25  27',
             '9   0    10    25  25',
             '10  1    10    25  35',
             '11  2    10    25  45']
    assert repr(df).split('\n') == lines

    two.run()
    df = two.returns.as_table(include_dep_args=True)
    lines = ['              x  offs  one|gain  one|offs     y',
             '0     [0, 1, 2]     0         1         0     5',
             '1     [0, 1, 2]    10         1         0    35',
             '2   [0, 10, 20]     0        10         0   500',
             '3   [0, 10, 20]    10        10         0   530',
             '4  [25, 26, 27]     0         1        25  2030',
             '5  [25, 26, 27]    10         1        25  2060',
             '6  [25, 35, 45]     0        10        25  3875',
             '7  [25, 35, 45]    10        10        25  3905']
    assert repr(df).split('\n') == lines

    three.add_dependency(two, squeeze=[[two, one.args.offs]],
                              auto_zip=False)
    three.args.x2.depends_on(two.returns.y)

    path = three.args._last_tasksweep.squeeze
    assert path == [[two.args.x, one.args.offs]]

    three.run()
    df = three.args.as_table(hide_const=True, include_dep_args=False)
    lines = ['              x1           x2',
             '0      [0, 1, 2]    [5, 2030]',
             '1    [0, 10, 20]    [5, 2030]',
             '2   [25, 26, 27]    [5, 2030]',
             '3   [25, 35, 45]    [5, 2030]',
             '4      [0, 1, 2]   [35, 2060]',
             '5    [0, 10, 20]   [35, 2060]',
             '6   [25, 26, 27]   [35, 2060]',
             '7   [25, 35, 45]   [35, 2060]',
             '8      [0, 1, 2]  [500, 3875]',
             '9    [0, 10, 20]  [500, 3875]',
             '10  [25, 26, 27]  [500, 3875]',
             '11  [25, 35, 45]  [500, 3875]',
             '12     [0, 1, 2]  [530, 3905]',
             '13   [0, 10, 20]  [530, 3905]',
             '14  [25, 26, 27]  [530, 3905]',
             '15  [25, 35, 45]  [530, 3905]']
    assert repr(df).split('\n') == lines


def test_4_with_auto_zip(task_setup):
    one, two, three, four = task_setup

    one.args.gain.iterate(1, 10)
    one.args.offs.iterate(0, 25)
    one.run()
    df = one.returns.as_table()
    lines = ['    x  gain  offs   y',
             '0   0     1     0   0',
             '1   1     1     0   1',
             '2   2     1     0   2',
             '3   0    10     0   0',
             '4   1    10     0  10',
             '5   2    10     0  20',
             '6   0     1    25  25',
             '7   1     1    25  26',
             '8   2     1    25  27',
             '9   0    10    25  25',
             '10  1    10    25  35',
             '11  2    10    25  45']
    assert repr(df).split('\n') == lines

    two.run()
    df = two.returns.as_table(include_dep_args=True)
    lines = ['              x  offs  one|gain  one|offs     y',
             '0     [0, 1, 2]     0         1         0     5',
             '1     [0, 1, 2]    10         1         0    35',
             '2   [0, 10, 20]     0        10         0   500',
             '3   [0, 10, 20]    10        10         0   530',
             '4  [25, 26, 27]     0         1        25  2030',
             '5  [25, 26, 27]    10         1        25  2060',
             '6  [25, 35, 45]     0        10        25  3875',
             '7  [25, 35, 45]    10        10        25  3905']
    assert repr(df).split('\n') == lines

    three.add_dependency(two, squeeze=[[two, one.args.offs]])
    three.args.x2.depends_on(two.returns.y)

    path = three.args._last_tasksweep.squeeze
    assert path == [[two.args.x, one.args.offs]]

    three.run()
    df = three.args.as_table(include_dep_args=False)

    lines = ['             x1           x2',
             '0     [0, 1, 2]    [5, 2030]',
             '1   [0, 10, 20]  [500, 3875]',
             '2  [25, 26, 27]    [5, 2030]',
             '3  [25, 35, 45]  [500, 3875]',
             '4     [0, 1, 2]   [35, 2060]',
             '5   [0, 10, 20]  [530, 3905]',
             '6  [25, 26, 27]   [35, 2060]',
             '7  [25, 35, 45]  [530, 3905]']
    assert repr(df).split('\n') == lines


def test_5(task_setup):
    one, two, three, four = task_setup

    one.args.gain.iterate(1, 10)
    one.args.offs.iterate(0, 25)
    one.run()
    df = one.returns.as_table()
    #~ print(df)
    lines = ['    x  gain  offs   y',
             '0   0     1     0   0',
             '1   1     1     0   1',
             '2   2     1     0   2',
             '3   0    10     0   0',
             '4   1    10     0  10',
             '5   2    10     0  20',
             '6   0     1    25  25',
             '7   1     1    25  26',
             '8   2     1    25  27',
             '9   0    10    25  25',
             '10  1    10    25  35',
             '11  2    10    25  45']
    assert repr(df).split('\n') == lines

    two.run()
    df = two.returns.as_table()  # better: include=[one.args.gain]
    #~ print(df)
    lines = ['              x  offs     y',
             '0     [0, 1, 2]     0     5',
             '1     [0, 1, 2]    10    35',
             '2   [0, 10, 20]     0   500',
             '3   [0, 10, 20]    10   530',
             '4  [25, 26, 27]     0  2030',
             '5  [25, 26, 27]    10  2060',
             '6  [25, 35, 45]     0  3875',
             '7  [25, 35, 45]    10  3905']
    assert repr(df).split('\n') == lines

    three.add_dependency(two, squeeze=[one.args.offs, two.args.offs],
                              auto_zip=False)
    three.args.x2.depends_on(two.returns.y)
    #~ three.args.x2.depends_on(two.args.x)

    path = three.args._last_tasksweep.squeeze
    assert path == [[two.args.x, one.args.offs], [two.args.offs]]

    three.run()
    df = three.args.as_table(include_dep_args=False)
    print(df)
    lines = ['             x1                      x2',
             '0     [0, 1, 2]     [5, 35, 2030, 2060]',
             '1   [0, 10, 20]     [5, 35, 2030, 2060]',
             '2  [25, 26, 27]     [5, 35, 2030, 2060]',
             '3  [25, 35, 45]     [5, 35, 2030, 2060]',
             '4     [0, 1, 2]  [500, 530, 3875, 3905]',
             '5   [0, 10, 20]  [500, 530, 3875, 3905]',
             '6  [25, 26, 27]  [500, 530, 3875, 3905]',
             '7  [25, 35, 45]  [500, 530, 3875, 3905]']
    assert repr(df).split('\n') == lines


def test_6(task_setup):
    one, two, three, four = task_setup

    one.args.gain.iterate(1, 10)
    one.args.offs.iterate(0, 25)
    one.run()

    two.run()

    three.add_dependency(two, squeeze=[one.args.offs, two.args.offs],
                              auto_zip=False)
    three.args.x2.depends_on(two.returns.y)

    three.run()
    df = three.args.as_table(include_dep_args=False)
    lines = ['             x1                      x2',
             '0     [0, 1, 2]     [5, 35, 2030, 2060]',
             '1   [0, 10, 20]     [5, 35, 2030, 2060]',
             '2  [25, 26, 27]     [5, 35, 2030, 2060]',
             '3  [25, 35, 45]     [5, 35, 2030, 2060]',
             '4     [0, 1, 2]  [500, 530, 3875, 3905]',
             '5   [0, 10, 20]  [500, 530, 3875, 3905]',
             '6  [25, 26, 27]  [500, 530, 3875, 3905]',
             '7  [25, 35, 45]  [500, 530, 3875, 3905]']
    assert repr(df).split('\n') == lines

    four.add_dependency(three, auto_zip=False)
    four.args.x1.depends_on(three.args.x1)
    four.args.x2.depends_on(three.args.x2)

    key_paths = four.args.x1._ptr._tasksweep.get_arg_paths()
    assert key_paths == [[three.args.x1, one.args.gain],
                         [three.args.x1, one.args.offs],
                         [three.args.x2, two.args.x, one.args.gain]]

    four.run()
    df = four.returns.as_table()
    lines = ['             x1                      x2     y',
             '0     [0, 1, 2]     [5, 35, 2030, 2060]  4133',
             '1   [0, 10, 20]     [5, 35, 2030, 2060]  4160',
             '2  [25, 26, 27]     [5, 35, 2030, 2060]  4208',
             '3  [25, 35, 45]     [5, 35, 2030, 2060]  4235',
             '4     [0, 1, 2]  [500, 530, 3875, 3905]  8813',
             '5   [0, 10, 20]  [500, 530, 3875, 3905]  8840',
             '6  [25, 26, 27]  [500, 530, 3875, 3905]  8888',
             '7  [25, 35, 45]  [500, 530, 3875, 3905]  8915']
    assert repr(df).split('\n') == lines
