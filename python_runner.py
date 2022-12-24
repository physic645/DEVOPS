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
nodes_layer_1_scenario_1      = [128]
epochs_scenario_1             = [100,200]

nodes_layer_1_scenario_2      = [64]
epochs_scenario_2             = [100,200]

searchname           = "heart"


f = open("demo_results.txt", "w")

f.write('nodes1st-epoch \t')
f.write('accuracy \t')
f.write('loss \t')
f.write('\n')
f.write('Scenario_1\n')

loss_all_scenario_1 = []
loss_all_scenario_2 = []
i=0

# Create a sequence of numbers of nodes 
# for number of first layer nodes here 10 until 30 with step 10 i.e 10,20,30
# product -> is Cartesian product =>
# takes all the possible combinatios of the input list
for args in product(nodes_layer_1_scenario_1,epochs_scenario_1):
    
    # Run the neural network with different first layer and epochs parameters
    # We create all the possible combinations here 10,1 / 10,2 / 20,1 / 20,2
    loss,accuracy = simple_NN(*args,searchname)   
    
    f.write(f'[{args[i]}-{args[i+1]}] \t\t\t\t')
    f.write(f'{accuracy*100:.3f} \t\t')
    f.write(f'{loss*100:.3f} \t')
    f.write('\n')
    
    
    print(f'[{args[i]}-{args[i+1]}] \t\t\t\t')
    print(f'{accuracy*100:.3f} \t\t')
    print(f'{loss*100:.3f} \t')
    print('\n')
    
    # Save all losses from the training into the loss_all list
    loss_all_scenario_1.append(loss)
    
    #f.write('Loss: %.2f\n' % (loss*100))
f.write('-- -- --\n')
i=0
f.write('Scenario_2\n')
# Create a sequence of numbers of nodes 
# for number of first layer nodes here 10 until 30 with step 10 i.e 10,20,30
# product -> is Cartesian product =>
# takes all the possible combinatios of the input list
for args in product(nodes_layer_1_scenario_2,epochs_scenario_2):
    
    # Run the neural network with different first layer and epochs parameters
    # We create all the possible combinations here 10,1 / 10,2 / 20,1 / 20,2
    loss,accuracy = simple_NN(*args,searchname)   
    
    f.write(f'[{args[i]}-{args[i+1]}] \t\t\t\t')
    f.write(f'{accuracy*100:.3f} \t\t')
    f.write(f'{loss*100:.3f} \t')
    f.write('\n')
    
    
    print(f'[{args[i]}-{args[i+1]}] \t\t\t\t')
    print(f'{accuracy*100:.3f} \t\t')
    print(f'{loss*100:.3f} \t')
    print('\n')
    
    # Save all losses from the training into the loss_all list
    loss_all_scenario_2.append(loss)
    
    #f.write('Loss: %.2f\n' % (loss*100))
f.write('-- -- --\n')

# Then here we just add the statistical analysis of the loss function

from scipy import stats

# Perform t-test for losses for the two groups
t_statistic, p_value = stats.ttest_ind(loss_all_scenario_1, loss_all_scenario_2)

# Print results
f.write(f'T-statistic: {t_statistic:.5f} \n')
f.write(f'P-value    : {p_value:.5f}\n')





end = time.time()
total = end-start

f.write(f'We trained the {p_kaggle.title} dataset made by {p_kaggle.creator} \n')
f.write(f'This neural network training took {total:.3f} seconds')
f.close()