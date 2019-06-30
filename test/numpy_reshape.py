# coding=utf-8

import numpy as np

# [1..9] の配列を 3×3 に変換
a = np.arange(1, 10)
b = np.reshape(-1, (3, 3))

print(a)
print(b)
# [1 2 3 4 5 6 7 8 9]
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]