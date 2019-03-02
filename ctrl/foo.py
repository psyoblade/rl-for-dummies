import random
import numpy as np
from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
from collections import deque

items = deque(maxlen=100)

for x in range(0, 100):
    sample = (x, x**2, x**3, 0)
    items.append(sample)


def debug(name, o):
    print(type(o))
    print(o)
    print('\n')


RandomSeeder.set_seed()

def test_transpose():
    samples = random.sample(items, 10)
    debug('list', samples)

    # slicing 0 column
    xxx = np.asarray(samples).transpose()
    t1, t2, t3, t4 = xxx
    debug('numpy', t4)


def foo():
    for x in range(1000):
        print(np.random.rand())

if __name__ == "__main__":
    foo()
