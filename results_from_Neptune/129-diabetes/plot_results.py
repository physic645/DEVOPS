# Plot two graphs together

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math

# Reading csv file into numpy array 
best        = np.loadtxt("C:/Users/a5138828/Documents/DEVOPS/results_from_Neptune/129-diabetes/best.csv", delimiter=",")
second_best = np.loadtxt("C:/Users/a5138828/Documents/DEVOPS/results_from_Neptune/129-diabetes/2b.csv", delimiter=",")
third_best  = np.loadtxt("C:/Users/a5138828/Documents/DEVOPS/results_from_Neptune/129-diabetes/3b.csv", delimiter=",")
fourth_best = np.loadtxt("C:/Users/a5138828/Documents/DEVOPS/results_from_Neptune/129-diabetes/4b.csv", delimiter=",")
fifth_best  = np.loadtxt("C:/Users/a5138828/Documents/DEVOPS/results_from_Neptune/129-diabetes/5b.csv", delimiter=",")

rows = 99

# Time dimension
t = best[1:rows,0]

best        = best[1:rows,2]
second_best = second_best[1:rows,2]
third_best  = third_best[1:rows,2]
fourth_best = fourth_best[1:rows,2]
fifth_best  = fifth_best[1:rows,2]

# Plotting 5 best models in the same plot
plt.plot(t, best,        color='b',      label='Proposed Polynomial NN')
plt.plot(t, second_best, color='r',      label='Second best', linestyle = '--', )
plt.plot(t, third_best,  color='orange', label='Third best',  linestyle = None,)
#plt.plot(t, fourth_best, color='c', label='Fourth best')
#plt.plot(t, fifth_best, color='y', label='Fifth best')

# Naming the x-axis, y-axis and the whole graph
plt.ylabel("Validation Accuracy")
plt.xlabel("Number of epochs")
plt.title("Performance of the 5 best models")
#plt.ylim(0.74)
# Adding legend, which helps us recognize the curve according to it's color
plt.legend(bbox_to_anchor =(0.5,-0.39), loc='lower center')
# plt.legend(bbox_to_anchor =(1,1), loc='lower center')
#plt.legend(loc='center right')
#plt.legend(bbox_to_anchor=(1.05, 0.0), loc='lower left', borderaxespad=0.)


# To load the display window
plt.show()
