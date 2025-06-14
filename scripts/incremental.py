import numpy as np
import matplotlib.pyplot as plt
from filters import Filter, plot_filters

# 1. Define initial filters
# Using dummy coefficients for demonstration
b1 = np.array([0.1, 0.2, 0.1])
a1 = np.array([1.0, -0.5, 0.2])
b2 = np.array([0.15, 0.25, 0.15])
a2 = np.array([1.0, -0.6, 0.3])

f1 = Filter(a=a1, b=b1, label="Filter A")
f2 = Filter(a=a2, b=b2, label="Filter B")

# 2. First plotting call
# The function creates the axes and returns them
fig, axes = plot_filters([f1, f2])

# 3. Compute a 3rd filter
b3 = np.array([0.05, 0.1, 0.05])
a3 = np.array([1.0, -0.4, 0.1])
f3 = Filter(a=a3, b=b3, label="Filter C")

# 4. Second plotting call
# We pass the existing 'axes' back into the function
# We can also pass extra plotting kwargs (like linestyle)
plot_filters([f3], axes=axes, linestyle='--', color='red')

plt.show()
print("Incremental plot generated successfully.")
