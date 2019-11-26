import numpy as np
import matplotlib.pyplot as plt


x = np.arange(2, 17, 1)
y1 = np.array([
    39636, 38412, 34719, 30339, 25376,
    21243, 20032, 13687, 11430, 9385,
    7542, 7088, 5862, 4706, 4566
])


plt.plot(x[:len(y1)], y1 / 40000 * 100, marker='o', markersize=3, color='k', label='conv_fc')
plt.title('Adversarial Performance')
plt.xlabel('epsilon (* 4/255)')
plt.ylabel('Accuracy on Train Dataset (%)')
plt.xlim(1, 17)
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.show()
