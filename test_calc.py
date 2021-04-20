import numpy as np
import math
depthImg = 0.90343
# depthImg = 0.91181
near = 0.2
far = 2.0

depth = far * near / (far - (far - near) * depthImg)

print(depth)


x, y = 0.205, 0.061666
a = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
height = math.sqrt(math.pow(depth, 2) - math.pow(a, 2))
print(height)

# angle = np.pi * 0.15

# roll = angle * -1 + (-np.pi * 0.5)

# if roll < -np.pi / 2:
#     roll += np.pi

# print(roll / np.pi)