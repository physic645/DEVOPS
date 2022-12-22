# python runner script

# Libraries

from simple_NN import simple_NN
from p_kaggle import p_kaggle
import p_kaggle

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

from itertools import product

import kaggle 
import time
import os

start = time.time()

# Define a list of parameters that we want to run
nodes_of_first_layer = [3]
epochs               = [1]
searchname           = "diabetes"


f = open("demo_results.txt", "w")

f.write('nodes1st-epoch \t')
f.write('accuracy \t')
f.write('loss \t')
f.write('\n')

i=0

# Create a sequence of numbers of nodes 
# for number of first layer nodes here 10 until 30 with step 10 i.e 10,20,30
# product -> is Cartesian product =>
# takes all the possible combinatios of the input list
for args in product(nodes_of_first_layer,epochs):
    
    # Run the neural network with different first layer and epochs parameters
    # We create all the possible combinations here 10,1 / 10,2 / 20,1 / 20,2
    loss,accuracy = simple_NN(*args,searchname)        
    f.write(f'[{args[i]}-{args[i+1]}] \t\t\t\t')
    f.write(f'{accuracy*100:.3f} \t\t')
    f.write(f'{loss*100:.3f} \t')
    f.write('\n')
    
    #f.write('Loss: %.2f\n' % (loss*100))
    
end = time.time()
total = end-start


#f.write(f'We trained the {title} dataset made by {creator} \n')
f.write(f'This neural network training took {total:.3f} seconds')
f.close()
    


