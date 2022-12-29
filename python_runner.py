# python runner script

# Libraries
from p_kaggle import p_kaggle
from run_NN   import run_NN 
from scipy    import stats
import time
import neptune.new as neptune


start = time.time()

run_nep_ai = neptune.init(
        project   ='k15redd22/MLOps',
        api_token ="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTA1M2VmOC0xZmUyLTQ4YzYtODdhYy0yNjRhY2E0NGM3YTAifQ==",
    )

# Define a list of parameters that we want to run
nodes_layer_1_scenario_1      = [64,128]
epochs_scenario_1             = [100]

nodes_layer_1_scenario_2      = [64,128]
epochs_scenario_2             = [100]

searchname                    = "diabetes"

# Call p_kaggle and download the working dataset once
X,y,input_shape = p_kaggle(searchname)

accuracy_1,loss_all_scenario_1 = run_NN(nodes_layer_1_scenario_1,epochs_scenario_1,X,y,input_shape,run_nep_ai)

accuracy_2,loss_all_scenario_2 = run_NN(nodes_layer_1_scenario_2,epochs_scenario_2,X,y,input_shape,run_nep_ai)

# Perform paired t-test for losses for the two scenarios
t_statistic, p_value = stats.ttest_rel(loss_all_scenario_1, loss_all_scenario_2)

# Print results
print(f'T-statistic: {t_statistic:.5f} \n')
print(f'P-value    : {p_value:.5f}     \n')


# Stops the connection to Neptune and synchronizes all data.
run_nep_ai.stop()

end = time.time()
total = end-start

import p_kaggle

print(f'We trained the {p_kaggle.title} dataset made by {p_kaggle.creator} \n')
print(f'This neural network training took {total:.3f} seconds')