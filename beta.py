import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

x_axis = np.linspace(0, 1, 1000)

# Beta Distribution
pdf_values1 = beta.pdf(x_axis, 14, 13)
pdf_values2 = beta.pdf(x_axis, 40, 37)

# plot
plt.figure(figsize=(8, 6))
plt.plot(x_axis, pdf_values1, label=r'$\alpha={}, \beta={}$'.format(14, 13))
plt.plot(x_axis, pdf_values2, label=r'$\alpha={}, \beta={}$'.format(40, 37))
plt.title('Beta Distribution')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.grid(True)
plt.show()

max_index = np.argmax(pdf_values2)
max_point = x_axis[max_index]
max_pdf = pdf_values2[max_index]
print("The MAP is", max_point)
