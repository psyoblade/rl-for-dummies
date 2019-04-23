import random
import numpy as np
from com.modulabs.ctrl.utils.SeedNumbers import RandomSeeder
from collections import deque

RandomSeeder.set_seed()
items = deque(maxlen=100)

for x in range(0, 100):
    sample = (x, x**2, x**3, 0)
    items.append(sample)


def debug(name, o):
    print(type(o))
    print(o)
    print('\n')


def test_transpose():
    samples = random.sample(items, 10)
    debug('list', samples)

    # slicing 0 column
    xxx = np.asarray(samples).transpose()
    t1, t2, t3, t4 = xxx
    debug('numpy', t4)

def test_random():
    for x in range(1000):
        print(np.random.rand())

def test_np_mean():
    list = []
    for x in range(100):
        list.append(x)
    mean = np.mean(list[-10:])
    print("{} = {}".format(mean, list[-10:]))


if __name__ == "__main__":
    test_np_mean()
