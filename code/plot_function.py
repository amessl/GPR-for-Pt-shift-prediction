
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = -np.exp(-0.4*x**2) + 1

squared = y**2

plt.plot(x, squared, color='orange')
plt.grid()
plt.show()


integral = np.trapz(squared, [-np.inf, np.inf])
print(integral)