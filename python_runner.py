# python runner script

# f.write(f'i = {i} \n')
from simple_NN import simple_NN
import time

start = time.time()

# Create a sequence of numbers of nodes for first layer 10 until 30 with step 5
f = open("demo_results.txt", "w")
f.write('i \t')
f.write('accuracy \t')
f.write('loss \t')
f.write('\n')
epochs = 10000
for i in range(10,31,10):
    
    # Run the neural network with different first layer parameters
    loss,accuracy = simple_NN(i,epochs)    
    f.write(f'{i} \t')
    f.write(f'{accuracy*100:.3f} \t')
    f.write(f'{loss*100:.3f} \t')
    f.write('\n')
    #f.write('Loss: %.2f\n' % (loss*100))
    
end = time.time()
total = end-start
f.write(f'This neural network training took {total:.3f} seconds')
f.close()
    


