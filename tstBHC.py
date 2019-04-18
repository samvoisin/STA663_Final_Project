import numpy as np

from "./classes/Leaf.py" import Leaf
from "./classes/Split.py" import Split

x1 = np.array([1, 2])
x2 = np.array([4, 3])
x3 = np.array([5, 6])


f1 = Leaf(x1)
f2 = Leaf(x2)
f3 = Leaf(x3)

s12 = Split(f1, f2)
s123 = Split(s12, f3)
print(s123.tier)