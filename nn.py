import random
from grad import Value

class Module:
    def __init__(self):
        pass

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True):
        super().__init__()
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def forward(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        kind = "ReLU" if self.nonlin else "Linear"
        return f"{kind}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin: int, nout: int, nonlin: bool = True):
        super().__init__()
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def forward(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return f"Layer([{', '.join(map(str, self.neurons))}])"


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]):
        super().__init__()
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP([\n  " + ",\n  ".join(str(layer) for layer in self.layers) + "\n])"
