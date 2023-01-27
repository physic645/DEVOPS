# Plot two graphs together

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math

# Reading csv file into numpy array 
best        = np.loadtxt("best.csv", delimiter=",")
second_best = np.loadtxt("second_best.csv", delimiter=",")

rows = 99

# Time dimension
t = best[1:rows,0]

best        = best[1:rows,2]
second_best = second_best[1:rows,2]

# Plotting both the curves simultaneously
plt.plot(t, best, color='b', label='Proposed Polynomial NN')
plt.plot(t, second_best, color='r', label='Best model without polynomial')

# Naming the x-axis, y-axis and the whole graph
plt.ylabel("Validation Accuracy")
plt.xlabel("Number of epochs")
plt.title("Proposed polynomial NN vs Best_model_without_polynomials")
#plt.ylim(0.80)
# Adding legend, which helps us recognize the curve according to it's color
plt.legend(bbox_to_anchor =(0.5,-0.39), loc='lower center')

# To load the display window
plt.show()
