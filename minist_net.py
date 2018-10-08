import numpy as np 
import matplotlib.pyplot as plt
from mnist_data import fetch_traingset, fetch_testingset
from gradientCheck import gradCheck
#######################################################################################################
# Task of this project: MNIST digits classification using your own neural network codes
#
# Requirements:
# The number of hidden layers must larger than 3
# At least 2 activation functions should be tested: ReLU, tanh, sigmoid, maxout, ELU etc
# Training on the full MNIST training set
# Reporting the classification accuracy on the whole testing set
# Project report should give your training error curves & your testing accuracies. 
# Your code should upload to github.com. A linker to the code should be given in your project report.
#
# Additional credits:
# Using BN in your networks
# The higher testing accuracy the better
#######################################################################################################

num_features = 784
num_class = 10

def load_data(num_train=50000, num_test=10000, num_val=10000, num_dev=5000):
    
    """
    - X (N, D)
    - y (N, ) 
    """
    
    X_train = fetch_traingset()['images'][0: num_train] 
    X_train = np.array(X_train).reshape(num_train, -1)

    y_train = fetch_traingset()['labels'][0:num_train]
    y_train = np.array(y_train)

    X_val = fetch_traingset()['images'][num_train: num_train+num_val]
    X_val = np.array(X_val).reshape(num_val, -1)

    y_val = fetch_traingset()['labels'][num_train: num_train+num_val]
    y_val = np.array(y_val)

    X_test = fetch_testingset()['images'][0: num_test]
    X_test = np.array(X_test).reshape(num_test, -1)

    y_test = fetch_testingset()['labels'][0:num_test]
    y_test = np.array(y_test)

    X_dev = fetch_traingset()['images'][0:num_dev] 
    X_dev = np.array(X_dev).reshape(num_dev, -1)
    
    y_dev = fetch_traingset()['labels'][0:num_dev]
    y_dev = np.array(y_dev)
    
    #print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # data normalization 
    mean_img = np.mean(X_train, axis=0)
    #std_img = np.std(X_train, axis=0)
    X_train -= mean_img
 
    X_test -= mean_img

    X_val -= mean_img

    X_dev -= mean_img
  
    
    return X_train, y_train, X_test, y_test, X_val, y_val, X_dev, y_dev


def explore_data():
    """
    PCA
    """
    pass

def relative_error(x, y):
	
	return np.max(np.abs(x - y) / np.maximum(1e-8, np.abs(x) + np.abs(y)))



class network(object):
    """
    3 hidden layers neural net
    """
    def __init__(self, h1, h2, h3, af='Relu'):
        """
        initialize params
        h1, h2, h3 -- number of hidden layers
        - W1 (num_features, h1)
        - b1 (h1, )
        """
        self.params = {}
        self.params['W1'] = np.sqrt(2 / (num_features + h1)) * np.random.rand(num_features, h1)
        self.params['b1'] = np.zeros(h1)
        
        self.params['W2'] = np.sqrt(2 / (h1 + h2)) * np.random.rand(h1, h2)
        self.params['b2'] = np.zeros(h2)
        
        self.params['W3'] = np.sqrt(2 / (h2 +h3)) * np.random.rand(h2, h3)
        self.params['b3'] = np.zeros(h3)    
        
        self.params['Wout'] = np.sqrt(2 / (h3 + num_class)) * np.random.rand(h3, num_class)
        self.params['bout'] = np.zeros(num_class)
        
        fLogit = lambda x : 1 / (1 + np.exp(-x))
        fRelu = lambda x : np.maximum(0, x)
        fTanh = lambda x : np.tanh(x)
        if af == 'Relu':
            self.actf = fRelu

        elif af == 'Tanh':
        	self.actf = fTanh

        elif af == 'Logit': 
            self.actf = fLogit

        else:
        	raise ValueError('the value of keyword af just can be ***"Relu" or "Tanh" or "Logit"***')

    
    def loss(self, X, y=None, reg=0.01):
        """"
        计算scores 
        （在给出y的情况下）计算loss, gradient
        - X (N, D)   
        - y (N, )
        - W1 (D, h1)
        - b1 (h1, )
        - Wout (h3, num_class)
        - scores (N, num_class)
        """
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        W3 = self.params['W3']
        b3 = self.params['b3']
        Wout = self.params['Wout']
        bout = self.params['bout']
       # print('b1', b1.shape)

        af = self.actf
        
        # forward pass
        margins1 = X.dot(W1) + b1  # (N ,h1)  (h1, )
       # print('mar', margins1.shape)
        l1 = af(margins1)
        margins2 = l1.dot(W2) + b2
        l2 = af(margins2)
        margins3 = l2.dot(W3) + b3
        l3 = af(margins3)
        scores = l3.dot(Wout) + bout  # (N, C)
        
        
        if y is None:
            return scores

        # compute softmax loss and use l2 regularization
        num_tr = X.shape[0]
        index_yi = (range(num_tr), y)
        
        subs = np.max(scores, axis=1).reshape(-1, 1)  # (N ,1)
        scores_exp = np.exp(scores -subs)  # (N, C)
        nume = scores_exp[index_yi]   # (N, )
        deno = np.sum(scores_exp, axis=1).reshape(-1, 1)   # (N,1)
        loss = np.sum(-np.log(np.maximum(nume.reshape(-1, 1) / deno, 1e-8))) / num_tr
        
        # L2 regularization
        loss += reg * (np.sum(np.square(W1)) + np.sum(np.square(W2))
                     + np.sum(np.square(W3)) + np.sum(np.square(Wout))) 

        # backward 
        div = scores_exp / deno  # (N, C)
        div[index_yi] -= 1  
        dWout = np.dot(l3.T, div) / num_tr
        dWout += 2 * reg * Wout   # (h3, C)
        dbout = np.sum(div, axis=0)/num_tr  #(C, )

        df3 = None 
        df2 = None 
        df1 = None

        if af(2) == 2 :   # Relu actf
            df3 = np.zeros_like(margins3)
            df3[margins3>0] = 1

            df2 = np.zeros_like(margins2)
            df2[margins2>0] = 1

            df1 = np.zeros_like(margins1)
            df1[margins1>0] = 1

        elif af(2) == np.tanh(2):
            df3 = 1 - np.square(l3)  # (N ,h3)    # first
            df2 = 1 - np.square(l2)  # (N ,h2)
            df1 = 1 - np.square(l1)  # (N ,h1)

        else :
            df3 = l3 * (1 - l3)
            df2 = l2 * (1 - l2)
            df1 = l1 * (1 - l1)
           
      
     
        dl3 = np.dot(div, Wout.T)   # (N, h3)
        pz3 = dl3 * df3
        dW3 = np.dot(l2.T, pz3) / num_tr  # (h2, h3)
        dW3 += 2 * reg * W3  
        db3 = np.sum(pz3, axis=0) / num_tr

        dm3 = dl3 * df3
        dl2 = np.dot(dm3, W3.T) # (N, h2)
        dm2 = dl2 * df2
        dW2 = np.dot(l1.T, dm2) / num_tr  
        dW2 += 2 * reg * W2
        db2 = np.sum(dm2, axis=0) / num_tr

        dm2 = dl2 * df2
        dl1 = np.dot(dm2, W2.T) # (N, h1)
        dm1= dl1 * df1   
        dW1 = np.dot(X.T, dm1) / num_tr #  (D, h1)
        dW1 += 2 * reg * W1
        db1 = np.sum(dm1, axis=0) / num_tr
        
        grad = dict(dW1= dW1, dW2=dW2, dW3=dW3, dWout=dWout,
                    db1=db1, db2=db2, db3=db3, dbout=dbout)
        
        return loss, grad



    def train(self, X, y, X_val, y_val, batch_size=500, iteration=1000, learning_rate=1e-4, reg=0.5, delay=0.95):
        """
       - batch_size
       - y (N, )
        """
        N = X.shape[0]
        loss_batch = []
        loss_history = []
        acc_tr = []
        acc_val = []
        for i in range(iteration):
            batch_index = np.random.choice(N, batch_size)
            X_batch = X[batch_index]
            y_batch = y[batch_index]

            loss, grad = self.loss(X_batch, y_batch, reg=reg)

            self.params['W1'] = self.params['W1'] - learning_rate * grad['dW1']
            self.params['W2'] = self.params['W2'] - learning_rate * grad['dW2']
            self.params['W3'] = self.params['W3'] - learning_rate * grad['dW3']
            self.params['Wout'] = self.params['Wout'] - learning_rate * grad['dWout']
            self.params['b1'] = self.params['b1'] - learning_rate * grad['db1']
            self.params['b2'] = self.params['b2'] - learning_rate * grad['db2']
            self.params['b3'] = self.params['b3'] - learning_rate * grad['db3']
            self.params['bout'] = self.params['bout'] - learning_rate * grad['dbout']

            loss_history.append(loss)

            if (i % 1000) == 0 :
                loss_batch.append(loss)
                print('{0}/{1} iterations-->the loss is {2}'.format(i, iteration, loss))
                

            if ((i*batch_size) % N) == 0:
                acc_val.append(self.predict(X_val, y_val, verbose=False))
                acc_tr.append(self.predict(X, y, verbose=False))
                learning_rate = delay * learning_rate
                
        return loss_history, acc_val, acc_tr

    def predict(self, X, y=None, verbose=True):
        """
        - X
        - y
        """
        scores = self.loss(X, reg=0.5)  # (N, C)
        pre_label = np.argmax(scores, axis=1)
        if y is None:
            return pre_label

        acc = np.mean(pre_label == y)
        if verbose is True:
            print('the accuracy on test dataset is {:.2%}'.format(acc))
        return acc
########################### Relu Tuning log ####### Xavier initialization #####
# order-h1---h2--h3--batch_size---learning_rate--reg---iteration---loss---test_acc (size: train-50000, test-1000)
#  01--100-100-100----200--------1e-3-----------0.005----60000----0.88----0.87
        
#  02--200-300-100----300--------1e-3-----------0.005----60000----1.38----0.77
        
#  03--200-300-100----300--------5e-3-----------0.001-----10000---1.25----0.7  
        
#  04--200-300-100----300--------8e-3-----------0.0001----10000---0.31----0.93
        
#  05--300-300-200----300--------9e-3-----------0.0001----10000---0.48----0.91
        
#  06--300-300-200----300--------1e-2-----------0.0005----10000---0.70----0.88
        
#  07--200-300-100----300--------9e-3-----------0.0001----10000---0.25----0.92

#  08--200-300-100----300--------1e-2-----------0.0001----10000---0.24----0.91

#  09--200-300-100----300--------1e-2(0.95)-----0.0001----10000---0.41----0.89

#  10--200-300-100----300--------2e-2(0.95)-----0.0001----10000---0.54----0.89
        
#  11--200-300-100----300-------1.5e-2(0.95)----0.0001----10000---0.55----0.81

#  12--200-300-100----300--------8e-3(0.95)-----0.0001----30000---0.50----0.89     

#  13--200-300-100----300--------8e-3(1)--------0.0001----30000---0.11----0.96
        
#  14--200-300-100----300--------9e-3(1)--------0.0001----30000---0.11----0.965
        
#  15--200-300-100----300--------9e-3(1)--------0.0003----30000---0.18----0.962
        
#  16--200-300-200----300--------9e-3(1)--------0.0003----30000---0.16----0.954
        
#  16--200-300-50-----300--------9e-3(1)--------0.0001----30000---0.09----0.966
        
#  16--200-200-50-----300--------9e-3(1)--------0.0001----30000---0.12----0.970
 
#  17--200-100-50-----300--------9e-3(1)--------0.0001----30000---0.13----0.959 
        
#  18--200-200-50-----300--------9e-3(1)--------0.0000----30000---0.09----0.9658
###############################################################################

############################# Tanh Tuning log #################################
# order-h1---h2--h3--batch_size---learning_rate--reg---iteration---loss---test_acc (size: train-50000, test-1000)
#  01--200-200-50-----300--------9e-3(1)--------0.0001----30000---0.9----0.8

X_train, y_train, X_test, y_test, X_val, y_val, X_dev, y_dev = load_data(num_train=50000, num_test=10000,
                                                                         num_val=10000, num_dev=10)
h1, h2, h3 = (100, 100, 50)
net = network(h1, h2, h3, af='Tanh')
loss_his, acc_val, acc_tr = net.train(X_train, y_train, X_val, y_val, batch_size=500, 
                                      learning_rate=2e-2, iteration=10000, reg=0.000, delay=1.0)
net.predict(X_test, y_test)

#   plot loss(iteration), acc_val & acc_tr (echo) 
plt.figure(1)
plt.subplot(121)
plt.plot(loss_his)

plt.subplot(122)
plt.plot(acc_tr, 'b', label='train_acc')
plt.plot(acc_val, 'g', label='validation_acc')
plt.legend()

# visulize the params
plt.figure(2)
params = net.params
W1 = params['W1'].reshape(num_features,-1)
W2 = params['W2'].reshape(h1, -1)
W3 = params['W3'].reshape(h2, -1)
Wout = params['Wout'].reshape(h3, -1)

plt.subplot(221)
plt.imshow(W1, cmap='gray')
plt.subplot(222)
plt.imshow(W2, cmap='gray')
plt.subplot(223)
plt.imshow(W3, cmap='gray')
plt.subplot(224)
plt.imshow(Wout, cmap='gray')

# =============================================================================
# 
# net1 = network(5, 5, 5, af='')
# grad_dev = net1.loss(X_dev[0:2], y_dev[0:2], reg=0.001)[1]
# f = lambda W : net1.loss(X_dev[0:2], y_dev[0:2], reg=0.001)[0]
# pa = ['W1', 'W2', 'W3', 'Wout', 'b1', 'b2', 'b3', 'bout']
# pa1 = ['W3', 'b3']
# for param in pa:
# 	grad_num = gradCheck(f, net1.params[param], verbose=False)
# 	print('d%s max relative error is %e' % (param, relative_error(grad_num, grad_dev['d'+param])))
# 
# 
# =============================================================================













