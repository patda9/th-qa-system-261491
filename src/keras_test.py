import numpy as np

list1 = [[1, 2, 3], [4, 5, 6]]
list2 = [[0, 0, 0]]
list3 = [[8, 8, 8]]
list1 = list1 + list2
list2 = list2 + list3
list1 = np.array(list1)

print(list2)
print(list1)