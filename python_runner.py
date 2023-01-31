# python runner script

# Libraries
from connect_with_kaggle                import connect_with_kaggle
from find_optimal_hyperparameters       import find_optimal_hyperparameters
from train_and_evaluate_hypermodel      import train_and_evaluate_hypermodel
#from train_with_optimal_hyperparameters_stat import train_with_optimal_hyperparameters_stat
from scipy                              import stats

from sklearn.model_selection            import train_test_split # split a dataset into train and test sets

import neptune.new as neptune
import time
from sklearn.preprocessing import LabelEncoder

start_program_time = time.time()

# ----------------------------------------------------------------------------
# ---- An automated machine learning pipeline of three steps ------------------
# ----------------------------------------------------------------------------


# Step 1: Connect_with_kaggle and download the working dataset onceee

searchname      = "surgical"
X,y,input_shape = connect_with_kaggle(searchname)


# If the dataset is larger than 2000 rows keep only 2000
if len(X) > 2000:
    X = X[:2000]

if len(y) > 2000:
    y = y[:2000]


# if the target variable is in string format convert to 0 or 1    
if type(y[0]) == str:
    le = LabelEncoder()
    y  = le.fit_transform(y)

print(len(X))
print(input_shape)
# split into train test sets
X, X_test, y, y_test = train_test_split(X, y, test_size=0.40)

# --------------------------- end step 1 ------------------------------------



# Step 2: Find hypermodels

start_hyper_time = time.time()

#times  = 100
#i      = 0


'''
for i in range(times):
    
    print(f'We are in the {i+1} iteration \n')
    
    t1,best,best_e,snd_best,best_without_pol = find_optimal_hyperparameters(X,y,input_shape,X_test,y_test)
    b = best.values
    
    #print(f'best proposed in {i+1} loop the proposed NN is: {b}')
    
    if 'h_2nd' in b.values():
        i = i + 1;

print(f'The hermittes 2nd order appears {i} in {times} times as a proposed activation. \n')        
'''




tuner1,best_hps,second_best,scenario_without_pol = find_optimal_hyperparameters(X,y,input_shape,X_test,y_test)    




# Print the 5 best models
print(f'The best        is: {best_hps.values} \n\n')
print(f'The second best is: {second_best.values} \n\n')
#print(f'The third_best  is: {third_best.values} \n\n')
#print(f'The fourth_best is: {fourth_best.values} \n\n')
#print(f'The fifth_best  is: {fifth_best.values} \n\n')


#print best hyperparamaters with 
# print(best_hps.get_config())  or 
# tuner.results_summary() --> shows the 10 best trials
# print(best_hps.values)

# Show the 7 best trials 
# print(f'\n{tuner1.results_summary(7)}\n')


end_hyper_time = time.time()

# --------------------------- end step 2 ----------------------------------




# Step 3: Train hypermodels for 100 epochs and evaluate on test data
# Connect with NeptuneAI

start_training_time_5_hypermodels = time.time()

epochs = 100


# Train the best model
loss_best, accuracy_best               = train_and_evaluate_hypermodel(tuner1,best_hps,epochs,X,y,X_test,y_test)

# Train the second_best_model
loss_best, accuracy_best               = train_and_evaluate_hypermodel(tuner1,second_best,epochs,X,y,X_test,y_test)

# Train the scenario_without_
loss_best, accuracy_best               = train_and_evaluate_hypermodel(tuner1,scenario_without_pol,epochs,X,y,X_test,y_test)


# Train the fourth_best_model
# loss_best, accuracy_best               = train_and_evaluate_hypermodel(tuner1,fourth_best,epochs,X,y,X_test,y_test)

# Train the fifth_best_model
# loss_best, accuracy_best               = train_and_evaluate_hypermodel(tuner1,fifth_best,epochs,X,y,X_test,y_test)



end_training_time_5_hypermodels = time.time()
# --------------------------- end step 3 ----------------------------------


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


import connect_with_kaggle
end_program_time   = time.time()

total_hyper_time                  = end_hyper_time                  - start_hyper_time
total_training_time_5_hypermodels = end_training_time_5_hypermodels - start_training_time_5_hypermodels
total_program_time                = end_program_time                - start_program_time

# Print associated times
print('\nThe associated times were: \n')

print(f'The total time for hypertuning the {connect_with_kaggle.title} dataset was: {total_hyper_time:.3f} seconds \n')

print(f'The total time for training the {connect_with_kaggle.title} dataset for the 3 best models for {epochs} epochs was: {total_training_time_5_hypermodels:.3f} seconds \n')

print(f'The whole process took total {total_program_time:.3f} seconds \n')

import connect_with_kaggle
print(f'\n We trained the {connect_with_kaggle.title} dataset made by {connect_with_kaggle.creator} \n')

# ----- END OF PROGRAM ----