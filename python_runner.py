# python runner script

from simple_NN import simple_NN

# Create a sequence of numbers of nodes for first layer 10 until 30 with step 5
f = open("demo_results.txt", "w")
for i in range(10,31,10):
    
    loss,accuracy = simple_NN(i)    
    f.write(f'i = {i} \n')
    f.write('Accuracy: %.2f\t' % (accuracy*100))
    f.write('Loss: %.2f\n' % (loss*100))
f.close()
    
    