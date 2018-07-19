def test_from_depend_tasks():
    #~ if __name__ == '__main__':
    from loopyplot import Task
    from collections import OrderedDict
    from numpy import array_equal


    @Task
    def one(x):
        y = x + 1
        return y
    one.args.x.sweep(0, 2)

    @Task
    def two(x, offs=0):
        y = x**2 + offs
        return y
    two.add_dependency(one, squeeze=[[one.args.x]])
    two.args.x.depends_on(one.returns.y)
    two.args.offs.iterate(0, 10)

    @Task
    def three(x1, x2, x3=3):
        y = x1 * sum(x2) + x3
        return y

    three.add_dependency(two, squeeze=[[two.args.offs]])
    three.args.x1.depends_on(two.args.offs)
    three.add_dependency(one, squeeze=[[one.args.x]])
    three.args.x2.depends_on(one.returns.y)
    #~ three.args.x3.depends_on(one.returns.y)

    @Task
    def four(x1, x2=0):
        y = x1 + x2
        return y
    four.add_dependency(three)
    four.args.x1.depends_on(three.returns.y)
    four.args.x2.sweep(1, 3)

    tasks = four.depend_tasks
    assert tasks == OrderedDict([(three, [four.args.x1]),
                                 (two,   [three.args.x1]),
                                 (one,   [three.args.x2, two.args.x])])

    path = four.complete_path(two.args.x)
    assert path == [four.args.x1, three.args.x1, two.args.x]

    path = four.complete_path(one.args.x)
    assert path == [four.args.x1, three.args.x2, one.args.x]

    path = four.complete_path([two, one.args.x])
    assert path == [four.args.x1, three.args.x1, two.args.x, one.args.x]

    one.run()
    df = one.returns.as_table()
    lines = ['   x  y',
             '0  0  1',
             '1  1  2',
             '2  2  3']
    assert repr(df).split('\n') == lines

    two.run()
    df = two.returns.as_table()
    lines = ['           x  offs             y',
             '0  [1, 2, 3]     0     [1, 4, 9]',
             '1  [1, 2, 3]    10  [11, 14, 19]']
    assert repr(df).split('\n') == lines

    three.run()
    df = three.returns.as_table()
    lines = ['        x1         x2  x3        y',
             '0  [0, 10]  [1, 2, 3]   3  [3, 63]']
    assert repr(df).split('\n') == lines

    four.run()
    df = four.returns.as_table()
    lines = ['        x1  x2        y',
             '0  [3, 63]   1  [4, 64]',
             '1  [3, 63]   2  [5, 65]',
             '2  [3, 63]   3  [6, 66]']
    assert repr(df).split('\n') == lines

    path = four.complete_path(three.args.x1)
    assert path == [four.args.x1, three.args.x1]

    values = four.get_value_from_path(path, 2)
    assert array_equal(values, [0, 10])

    path = four.complete_path(three.args.x2)
    values = four.get_value_from_path(path, 2)
    assert array_equal(values, [1, 2, 3])

    path = four.complete_path(two.args.offs)
    values = four.get_value_from_path(path, 2)
    assert array_equal(values, [0, 10])

    path = four.complete_path(two.returns.y)
    values = four.get_value_from_path(path, 2)
    assert array_equal(values[0], [ 1,  4,  9])
    assert array_equal(values[1], [11, 14, 19])

    path = four.complete_path(two.args.x)
    values = four.get_value_from_path(path, 2)
    assert array_equal(values[0], [1, 2, 3])
    assert array_equal(values[1], [1, 2, 3])

    path = four.complete_path([two, one.args.x])
    values = four.get_value_from_path(path, [1, 2])
    assert values == [[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]]

