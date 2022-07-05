class DummyModel:

    def __init__(self, a:float=0.5, b:float=2, c:float=1) -> None:
        self.a = a
        self.b = b
        self.c = c

    def predict(self, x) -> float:
        return self.a * x * x + self.b * x + self.c

    def __call__(self, x) -> float:
        return self.predict(x)
