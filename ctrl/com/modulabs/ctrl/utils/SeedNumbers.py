#!/usr/bin/env python
# -*- coding:utf-8 -*-

# 매번 같은 숫자를 내어준다는 의미가 아니라, 시드 함수 호출 이후 항상 같은 랜덤 숫자를 반환합니다
# <a href="https://machinelearningmastery.com/reproducible-results-neural-networks-keras/">Reproducible_Keras_Vars</a>

from numpy.random import seed  # 넘피 변수도 랜덤 시드를 활용합니다
from tensorflow import set_random_seed  # 텐서 플로우의 경우에도 마찬가지로 시드 변수를 제공하면 동일한 효과를 가질 수 있습니다

random_seed = 1047104523


class RandomSeeder:
    global random_seed

    def __init__(self):
        self.set_seed()

    @staticmethod
    def set_seed(seed_number=random_seed):
        seed(seed_number)
        set_random_seed(seed_number)
        print("set_random_seed({})".format(seed_number))

    @staticmethod
    def reset():
        seed()
        set_random_seed(0)

