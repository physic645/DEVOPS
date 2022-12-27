# python runner script

# Libraries
from p_kaggle import p_kaggle
from run_NN   import run_NN 
from scipy    import stats
import time

start = time.time()

# Define a list of parameters that we want to run
nodes_layer_1_scenario_1      = [128]
epochs_scenario_1             = [10000]

nodes_layer_1_scenario_2      = [128]
epochs_scenario_2             = [1000000]

searchname                    = "heart"

# Call p_kaggle and download the working dataset once
X,y,input_shape = p_kaggle(searchname)


accuracy_1,loss_all_scenario_1 = run_NN(nodes_layer_1_scenario_1,epochs_scenario_1,X,y,input_shape)

accuracy_2,loss_all_scenario_2 = run_NN(nodes_layer_1_scenario_2,epochs_scenario_2,X,y,input_shape)


# Perform paired t-test for losses for the two scenarios
t_statistic, p_value = stats.ttest_rel(loss_all_scenario_1, loss_all_scenario_2)


# Print results
print(f'T-statistic: {t_statistic:.5f} \n')
print(f'P-value    : {p_value:.5f}     \n')


end = time.time()
total = end-start

import p_kaggle

print(f'We trained the {p_kaggle.title} dataset made by {p_kaggle.creator} \n')
print(f'This neural network training took {total:.3f} seconds')