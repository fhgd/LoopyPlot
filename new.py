class State:
    def __init__(self, init=None):
        self._init = init
        self.set(init)

    def get(self):
        return self._state

    def set(self, value):
        self._state = value
        self._is_initialized = True

    def reset(self):
        self.set(self._init)


class System:
    def __init__(self):
        self.state = State(0)

    def state_next(self, state):
        return state + 1

    def __len__(self):
        return 10

    def is_running(self):
        return 0 <= self.state._state < len(self) - 1

    def is_finished(self):
        return not self.is_running()

    def is_interrupted(self):
        state = self.state.get()
        return state % 3 == 0

    def __next__(self):
        if self.is_finished():
            raise StopIteration
        elif self.state._is_initialized:
            self.state._is_initialized = False
        else:
            self.state._state = self.state_next(self.state._state)
            if self.is_interrupted():
                self.state._is_initialized = True
                raise StopIteration
        return self.state.get()

    def __iter__(self):
        return self

    def as_list(self):
        self.reset()
        return list(self)

    def reset(self):
        self.state.reset()


if __name__ == '__main__':
    s = System()

    for idx in range(10):
        print(list(s))
