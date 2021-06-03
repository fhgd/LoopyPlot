class Loop:
    """
    lp = Loop(2, 4, 6, 8)

    def myfunc(lp):
        for n in lp:
            print(n)

    for _ in range(4):
        myfunc(lp)
        myfunc(lp)
        lp.next()

    """
    def __init__(self, *seq):
        self.seq = seq
        self._idx = 0
        self._idx_next = 0

    def __len__(self):
        return len(self.seq)

    def next_value(self):
        if self._is_initialized:
            self._is_initialized = False
        else:
            if self.is_running():
                self.next_idx()
            else:
                raise StopIteration
        return self.value

    def __next__(self):
        if self._idx == self._idx_next:
            self._idx_next += 1
            return self.seq[self._idx]
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def get(self):
        if self._idx == self._idx_next:
            self._idx_next += 1
        return self.seq[self._idx]

    value = property(get)

    def next(self):
        self._idx = self._idx_next

    def is_running(self):
        return 0 <= self._idx < len(self) - 1

    def reset(self):
        self._idx = 0
        self._idx_next = 0


class TestManager:
    def __init__(self):
        self._func = None  # current functiond
        self._funcs = {}   # func: [Loop, Loop, ...]

    def loop(self, *args):
        return self._funcs.setdefault(self._func, Loop(*args))

    def run(self, func):
        self._func = func
        self._funcs.pop(func, None)
        data = func(self)
        loop = self._funcs.get(func, None)
        if loop:
            data = [data]
            while loop.is_running():
                loop.next()
                data.append(func(self))
        return data


tm = TestManager()

def myloop(tm):
    for n in tm.loop(2, 4, 6, 8):
        print(n)
    return n
tm.run(myloop)


def myvar(tm):
    idx = tm.loop(1, 3, 5, 7).value
    print(idx)
    return idx
tm.run(myvar)
