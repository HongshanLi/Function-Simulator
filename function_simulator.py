import numpy as np
import tensorflow as tf
import sys
import time

start_time = time.time()

Train_Data_Size = []
for i in range(1):
   Train_Data_Size.append(5000)

Test_Data_Size = []
for i in range(1):
   Test_Data_Size.append(5000)
    
Iterations = []
for i in range(1):
   Iterations.append(100)

    
Depth = []
for i in range(1):
   Depth.append(3)


if len(Train_Data_Size) != len(Test_Data_Size) or len(Train_Data_Size) != len(Iterations) or len(Train_Data_Size) != len(Depth):
    print(len(Train_Data_Size))
    print(len(Test_Data_Size))
    print(len(Depth))
    sys.exit()

    
sess = tf.InteractiveSession()
input_vector_length = int(10)
batch_size = int(100)
function_domain = [0, 1]

x = tf.placeholder(tf.float32, shape = [None,input_vector_length])
y_ = tf.placeholder(tf.float32, shape = [None, 1])
    
class Data(object):
        
    def __init__(self, variable, label):
        self.variable = variable
        self.label = label
        
    def next_batch(self, batch_size):
        T = []
        L = []
        for i in np.random.randint(0, high = len(self.variable), size = batch_size):
            t = self.variable[i]
            T.append(t)
            l = self.label[i]
            L.append(l)
        return [T, L]




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


mu = tf.placeholder(tf.float32, shape = [None, input_vector_length])
def function(mu):                                               
    return 0.0025*tf.pow(tf.reduce_sum(mu, 1), 3) + 0.25*tf.pow(tf.reduce_sum(mu, 1), 2) + tf.reduce_sum(mu, 1)
    

results = open('results.tex', 'a')
results.write(r'\begin{tabular}{|c|c|c|c|c|}'+'\n')
results.write('\hline Train data size & Test data size & Iterations & Depth & Test error' + r'\\' + '\n')
results.close()

for train_data_size, test_data_size, iterations, depth in zip(Train_Data_Size, Test_Data_Size, Iterations, Depth):
   
    
    w = tf.placeholder(tf.float32, shape = [train_data_size, input_vector_length])
    z = tf.placeholder(tf.float32, shape = [test_data_size, input_vector_length])    
    sample_input = (function_domain[1] - function_domain[0])*np.random.rand(train_data_size, input_vector_length)
    sample_output = tf.Session().run(function(w), feed_dict = {w : sample_input})
    train_data = Data(sample_input, sample_output)
    
    sample_input = (function_domain[1] - function_domain[0])*np.random.rand(test_data_size, input_vector_length)
    - function_domain[0]
    sample_output = tf.Session().run(function(z), feed_dict = {z : sample_input})
    test_data = Data(sample_input, sample_output)

    weight_matrices = []
    bias_matrices = []
    for i in range(depth):
        W = weight_variable([input_vector_length, input_vector_length])
        weight_matrices.append(W)
        B = bias_variable([input_vector_length])
        bias_matrices.append(B)
        
    W_output = weight_variable([input_vector_length, 1])
    weight_matrices.append(W_output)
    
    B_output = bias_variable([1])
    bias_matrices.append(B_output)
    
    layer_output = []
    for i in range(depth):
        if i == 0:
            y = x
        y = tf.sigmoid(tf.matmul(y, weight_matrices[i]) + bias_matrices[i])
        layer_output.append(y)
    
    y_output = tf.matmul(y, weight_matrices[depth]) + bias_matrices[depth]
    
    
    cost_function = tf.reduce_mean(tf.square(tf.sub(y_output, y_)))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost_function)
    error = tf.reduce_mean(tf.square(y_output - y_))

    sess.run(tf.initialize_all_variables())
    for i in range(iterations):
        batch = train_data.next_batch(batch_size)
        train_step.run(feed_dict = {x: np.reshape(batch[0], [batch_size, input_vector_length]), y_ : np.reshape(batch[1], [batch_size, 1])})

    test_error = error.eval(feed_dict = {x : np.reshape(test_data.variable, [len(test_data.variable), input_vector_length]), 
                                         y_ : np.reshape(test_data.label, [len(test_data.label), 1])})

    print('train data size:%d, test data size:%d, iterations:%d, depth:%d, test error:%g' %
          (train_data_size, test_data_size, iterations, depth, test_error))
    
    

 
    results = open('results.tex', 'a')
    train_data_size = str(train_data_size)
    test_data_size = str(test_data_size)
    iterations = str(iterations)
    depth = str(depth)
    test_error = str(test_error)
    results.write('\hline ' + train_data_size + ' & ' + test_data_size + ' & ' + iterations + ' & ' + depth + ' & ' + test_error + r'\\'+'\n')
    
results.write('\hline \n')
results.write(r'\end{tabular}')
results.close()
    

print('training time is: %f' % (time.time() - start_time))
        
