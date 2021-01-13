#把0-1之间的数据映射到-1到1
def change_range(image,label):
    return 2*image-1,label

import random
import numpy as np

a = np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
np.random.shuffle(a)
print(a)
b = np.array([[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
random.shuffle(b)
print(b)
#np.random.shuffle改变元素顺序且不会改变元素