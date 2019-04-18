import numpy as np
import bhc as bhc

x1 = np.array([1, 2])
x2 = np.array([4, 3])
x3 = np.array([5, 6])

a = 10

f1 = bhc.leaf.Leaf(x1, a)
f2 = bhc.leaf.Leaf(x2, a)
f3 = bhc.leaf.Leaf(x3, a)

s12 = bhc.split.Split(f1, f2)
s123 = bhc.split.Split(s12, f3)
print(s123.tier)