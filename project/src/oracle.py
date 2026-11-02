class Oracle:
    """Обгортка цільової для розрахунку звернення"""

    def __init__(self, func):
        self.func = func
        self.n_f = 0
        self.history = []

    def value(self, x):
        """Обчислення ф-ції, +1"""
        self.n_f += 1
        v = float(self.func(x))
        self.history.append((x, v))
        return v

    def peek(self, x):
        """Обчисленя тільки для графіків"""
        return float(self.func(x))

    def reset(self):
        """Обнулити для наступного"""
        self.n_f = 0
        self.history.clear()