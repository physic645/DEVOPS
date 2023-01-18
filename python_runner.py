# python runner script

# Libraries
from connect_with_kaggle                import connect_with_kaggle
from find_optimal_hyperparameters       import find_optimal_hyperparameters
from train_with_optimal_hyperparameters import train_with_optimal_hyperparameters
#from train_with_optimal_hyperparameters_stat import train_with_optimal_hyperparameters_stat
from scipy                              import stats

from sklearn.model_selection            import train_test_split # split a dataset into train and test sets

import neptune.new as neptune
import time

start = time.time()

'''
# Initiate connection with Neptune AI for monitoring
run = neptune.init_run(
        project   ='k15redd22/MLOps',
        api_token ="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0NTA1M2VmOC0xZmUyLTQ4YzYtODdhYy0yNjRhY2E0NGM3YTAifQ==",
    )
'''

# ----------------------------------------------------------------------------
# ---- An automated machine learning pipeline of three steps -----------------

# Step:1
# Connect_with_kaggle and download the working dataset once

searchname      = "diabetes"
X,y,input_shape = connect_with_kaggle(searchname)

# split into train test sets
X, X_test, y, y_test = train_test_split(X, y, test_size=0.40)



# Step:2
# Find the optimal hyperparameter for the speficic dataset
i = 0
times = 100

for i in range(times):
    
    print(f'We are in the {i+1} iteration \n')
    
    tuner1,best_hps,best_epoch,second_best,scenario_without_pol = find_optimal_hyperparameters(X,y,input_shape,X_test,y_test)
    b = best_hps.values
    
    if 'h_2nd' in b.values():
        i = i + 1;
        
    
    
print(f'The hermittes 2nd order appears {i} in {times} times as a proposed activation. \n')
print(f'The proposed NN is: {b} \n\n')
print(f'Best model without polynomial is {scenario_without_pol.values}\n')
#print best hyperparamaters with 
# print(best_hps.get_config())  or 
# tuner.results_summary() --> shows the 10 best trials
# print(best_hps.values)

# Show the 7 best trials 
# print(f'\n{tuner1.results_summary(7)}\n')




# Step:3
# Train the hypermodel with optimal hyperparamters and evaluate on test data
# Connect with NeptuneAI

# Train the best model
loss_best, accuracy_best = train_with_optimal_hyperparameters(tuner1,best_hps,best_epoch,X,y,X_test,y_test)

# Train the best that not contains the polynomial
loss_without_pol, accuracy_without_pol = train_with_optimal_hyperparameters(tuner1,scenario_without_pol,best_epoch,X,y,X_test,y_test)

'''
# previous comment line
# Step 4: Statistics
    
loss_list_1     = []
accuracy_list_1 = []

loss_list_2     = []
accuracy_list_2 = []


loss_list_3     = []
accuracy_list_3 = []


# Loop 50 times for statistics / 50 values for loss and accuracy

for i in range(100):
    loss, accuracy = train_with_optimal_hyperparameters_stat(tuner1,best_hps,best_epoch,X,y,X_test,y_test)
    loss_list_1.append(loss)
    accuracy_list_1.append(accuracy)


#for i in range(10):
#    loss, accuracy = train_with_optimal_hyperparameters(tuner1,second_best,best_epoch,X,y,X_test,y_test)
#    loss_list_2.append(loss)
#   accuracy_list_2.append(accuracy)    


# loss and accuracy for a model that does not contain polynomial

for i in range(100):
    loss, accuracy = train_with_optimal_hyperparameters_stat(tuner1,scenario_without_pol,best_epoch,X,y,X_test,y_test)
    loss_list_3.append(loss)
    accuracy_list_3.append(accuracy)


# Perform paired t-test for losses for the two scenarios
t_stat_loss, p_value_loss = stats.ttest_rel(loss_list_1, loss_list_3)
print(f'Statistics for losses: t = {t_stat_loss}, p_value = {p_value_loss}\n')

t_stat_acc, p_value_acc = stats.ttest_rel(accuracy_list_1, accuracy_list_3)
print(f'Statistics for accuracy t = {t_stat_acc}, p_value = {p_value_acc}\n')
    

# ****************************************************************************
# ****************************************************************************
# ****************************************************************************

print(f'Best_model is {best_hps.values}\n')

print(f'Best model without polynomial is {scenario_without_pol.values}\n')

# Stops the connection to Neptune and synchronizes all data.
#run.stop()
'''

end   = time.time()
total = end-start

import connect_with_kaggle

print(f'\n We trained the {connect_with_kaggle.title} dataset made by {connect_with_kaggle.creator} \n')
print(f'This neural network training took {total:.3f} seconds')
