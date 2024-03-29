# runner of the neural network

from itertools import product
from simple_NN import simple_NN


# Create a sequence of numbers of nodes 
# for number of first layer nodes here 10 until 30 with step 10 i.e 10,20,30
# product -> is Cartesian product =>
# takes all the possible combinatios of the input list



def run_NN(nodes_layer_1_scenario_i,epochs_scenario_i,X,y,input_shape,run_nep_ai):
    
    accuracy_all_scenario_i = []
    loss_all_scenario_i     = []
    

    for args in product(nodes_layer_1_scenario_i,epochs_scenario_i):
               
        loss,accuracy = simple_NN(*args,X,y,input_shape,run_nep_ai)   
        
        accuracy_all_scenario_i.append(accuracy)
        loss_all_scenario_i.append(loss)
        
        print(f'Accuracy list -> {accuracy_all_scenario_i}\n')
        print(f'Loss list -> {loss_all_scenario_i} \n')
            
    # Return the list
    return accuracy_all_scenario_i,loss_all_scenario_i       