# Plot two graphs together

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math

# Reading csv file into numpy array 
best        = np.loadtxt("best.csv", delimiter=",")
second_best = np.loadtxt("2b.csv", delimiter=",")
third_best  = np.loadtxt("3b.csv", delimiter=",")


rows = 99

# Time dimension
t = best[1:rows,0]

best        = best[1:rows,2]
second_best = second_best[1:rows,2]
third_best  = third_best[1:rows,2]


# Plotting 5 best models in the same plot
plt.plot(t, best,color='b',
         label='NN_1: {30:Hermite, 10:ReLU, 20:Hermite, 5:ReLU, 5:ReLU}',linestyle =None )
plt.plot(t, second_best, color='r', 
         label='NN_2: {5:Hermite 30:ReLU,, 30:Hermite, 5:ReLU, 5:Hermite}',  linestyle = '--', )
plt.plot(t, third_best,  color='g', 
         label='NN_3: {30:ReLU, 10:ReLu, 30:ReLU, 5:ReLU, 5:ReLU,}',  linestyle = 'dotted',)


# Naming the x-axis, y-axis and the whole graph
plt.ylabel("Validation Accuracy")
plt.xlabel("Number of epochs")
title = plt.title("Room Occupancy: Performance of the 3 best models on the test data")

plt.ylim(0.70)

# Adding legend, which helps us recognize the curve according to it's color

# plt.legend(bbox_to_anchor =(0.5,-0.45), loc='lower center')
# plt.legend(bbox_to_anchor =(1,1), loc='lower center')
leg = plt.legend(loc='lower right',prop={'size': 9})
leg.get_frame().set_facecolor('whitesmoke')
#leg.set_facecolor('lightgray')
#plt.legend(bbox_to_anchor=(1.05, 0.0), loc='lower left', borderaxespad=0.)


# To load the display window
plt.show()
