import matplotlib.pyplot as plt
from imageio import imread



name = 'depth_jacquard3'
img = imread(f'test_input/{name}.tiff')
plt.imshow(img)
plt.colorbar(label='Pixel value')
plt.title('Depth image')
plt.show()