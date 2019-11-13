import numpy as np
import matplotlib.pyplot as plt


x = [
    1, 2, 3, 4, 5, 6, 7,
    8, 9, 11, 12, 13, 14, 15,
    17, 18, 19, 20, 21, 22, 23,
    26, 34, 35, 40, 48, 60, 83,
    102, 106, 113, 123, 133,
]

y1 = [
    21.32, 34.43, 41.56, 50.50, 52.90, 55.90, 57.30,
    59.49, 62.47, 65.41, 66.46, 67.59, 69.17, 70.12,
    70.86, 70.88, 71.57, 73.19, 73.35, 74.39, 74.21,
    75.28, 75.86, 76.25, 77.62, 76.98, 77.14, 77.44,
    77.82, 77.68, 77.56, 77.85, 77.84,
]

y2 = [
    15.95, 24.42, 30.64, 31.48, 32.70, 34.71, 36.17,
    38.72, 40.89, 41.81, 41.78, 42.04, 42.75, 43.48,
    45.22, 44.55, 44.52, 46.26, 47.04, 46.18, 46.49,
    45.53, 44.63, 44.77, 43.55, 42.29, 42.05, 41.54,
    41.73, 42.26, 42.37, 42.78, 42.77,
]

y3 = [
    16.62, 25.57, 31.84, 33.16, 34.60, 36.68, 38.10,
    40.88, 43.26, 43.69, 43.99, 44.03, 44.79, 45.56,
    46.82, 46.17, 46.62, 47.90, 48.66, 47.76, 48.33,
    46.88, 45.29, 45.38, 43.91, 41.99, 41.57, 40.50,
    40.58, 41.09, 40.85, 41.49, 41.32,
]


plt.plot(x[:len(y1)], y1, marker='x', color='k', label='Original Acc.')
plt.plot(x[:len(y2)], y2, marker='x', label='FGSM-Attacked Acc.')
plt.plot(x[:len(y3)], y3, marker='x', label='PGD-Attacked Acc.')
plt.title('PGD-Training #1')
plt.xlabel('# of epochs')
plt.ylabel('Accuracy on Test Dataset (%)')
plt.legend()
plt.show()
