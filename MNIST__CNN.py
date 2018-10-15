import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import time
tic = time.time()
#from mnist_dataset import MnistDataset
use_cuda = torch.cuda.is_available()

root = './data'
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

class LeNet(nn.Module):
    """ 
     structure: conv1(1, 20, 5, 1) ->  relu -> pooling(2, 2)
    -> conv2(20, 50, 5, 1) -> relu -> pooling 
    -> fc1(4*4*50， 500)  -> dropout -> fc2(500, 10)

    """
    def __init__(self):
        """"""
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5 , 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x):
        """
        - x : (1, 28, 28)
        """
        x = self.conv1(x).clamp(min=0)  # (20, 24, 24)
        x = F.max_pool2d(x, 2, 2) # (20, 12, 12)
        x = self.conv2(x).clamp(min=0) # (50, 8, 8)
        x = F.max_pool2d(x, 2, 2) # (50, 4, 4)
        x = x.view(-1, 50 * 4 * 4)
        x = self.fc1(x).clamp(min=0)
       # x = F.dropout(x, p=0.1)
        x = self.fc2(x)
        
        return x

class LeNet_BN(nn.Module):
    """ 
     structure: conv1(1, 20, 5, 1) ->  relu -> pooling(2, 2)
    -> conv2(20, 50, 5, 1) -> relu -> pooling 
    -> fc1(4*4*50， 500)   -> fc2(500, 10)

    """
    def __init__(self):
        """"""
        super(LeNet_BN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5 , 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(50)
    
    def forward(self, x):
        """
        - x : (1, 28, 28)
        """
        x = self.conv1(x).clamp(min=0)  # (20, 24, 24)
        x = self.bn1(x)
        x = F.max_pool2d(x, 2, 2) # (20, 12, 12)
        x = self.conv2(x).clamp(min=0) # (50, 8, 8)
        x = self.bn2(x)
        x = F.max_pool2d(x, 2, 2) # (50, 4, 4)      
        x = x.view(-1, 50 * 4 * 4)     
        x = self.fc1(x).clamp(min=0)
        x = self.fc2(x)
        
        return x   

class LeNet_leaky(nn.Module):
    """ 
     structure: conv1(1, 20, 5, 1) ->  leaky_relu -> pooling(2, 2)
    -> conv2(20, 50, 5, 1) -> leaky_relu -> pooling 
    -> fc1(4*4*50， 500)  -> fc2(500, 10)

    """
    def __init__(self):
        """"""
        super(LeNet_leaky, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5 , 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50*4*4, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x):
        """
        - x : (1, 28, 28)
        """
        x = F.leaky_relu(self.conv1(x))  # (20, 24, 24)
        x = F.max_pool2d(x, 2, 2) # (20, 12, 12)
        x = F.leaky_relu(self.conv2(x)) # (50, 8, 8)
        x = F.max_pool2d(x, 2, 2) # (50, 4, 4)
        x = x.view(-1, 50 * 4 * 4)
        x = F.leaky_relu(self.fc1(x))
       # x = F.dropout(x, p=0.1)
        x = self.fc2(x)
        
        return x
class LeNet_3(nn.Module):
    """ 
     structure: conv1(1, 20, 5, 1) ->  relu -> pooling(2, 2)
    -> conv2(20, 50, 5, 1) -> relu -> pooling 
    -> fc1(4*4*50， 500)  -> dropout -> fc2(500, 10)

    """
    def __init__(self):
        """"""
        super(LeNet_3, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3 , 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1)
        self.conv4 = nn.Conv2d(50, 20, 3, 1)
        self.fc1 = nn.Linear(20*4*4, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x):
        """
        - x : (1, 28, 28)
        """
        x = self.conv1(x).clamp(min=0)  
        x = self.conv2(x).clamp(min=0) 
        x = F.max_pool2d(x, 2, 2) 
        x = self.conv3(x).clamp(min=0) 
        x = self.conv4(x).clamp(min=0) 
        x = F.max_pool2d(x, 2, 2) 
        x = x.view(-1, 20 * 4 * 4)
        x = self.fc1(x).clamp(min=0)
        x = self.fc2(x)
        
        return x


    
class Net():
    def __init__(self, inst):
        """
        - inst : LeNet
        """
        self.inst = inst
    
    def train(self, train_loader, val_dataloader, SGD_update, num_epoch=10):
        """
        - dataset   [ tensor(N, 1, H, W) float32 , tensor(N) in64 ]
        - batch_size
        - learning_rate
        - updata_rule
        - epoch
        """
        loss_history = []
        acc_train = []
        acc_val = []
        moving_ave = 0.0
        
        criterion = nn.CrossEntropyLoss()
        model = self.inst
        if use_cuda:
            model = model.cuda()
        #if update_rule == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), **SGD_update) 
        for i_epoch in range(num_epoch):
            for i_batch, data in enumerate(train_loader):
              
                X_batch = data[0]
                y_batch = data[1]
                
                if use_cuda:
                    X_batch = X_batch.cuda()
                    y_batch = y_batch.cuda()
                
               # print(X_batch.shape, X_batch.dtype)
                scores = model(X_batch)    
                loss = criterion(scores, y_batch)  # (N, C)  (N) -> [1]    
                moving_ave = 0.9 * moving_ave + 0.1 * loss.item()
                loss_history.append(moving_ave)
                optimizer.zero_grad()
                loss.backward()       
                optimizer.step()
            print('>>> epoch:', i_epoch, 'finished -->', moving_ave)
                
            acc_train.append(self.predict(train_loader, verbose=False))
            acc_val.append(self.predict(val_loader, verbose=False))
                       
        return loss_history, acc_train, acc_val
                      
    
    def predict(self, dataloader, verbose=True):
        acc_history = []
        for i_batch, data in enumerate(dataloader):
            X = data[0]
            y = data[1]
            if use_cuda:
                X = X.cuda()
                y = y.cuda()
            scores = self.inst(X)
            y_pre = scores.argmax(dim=1)
            acc = torch.sum(y==y_pre).item() / y.shape[0]
            acc_history.append(acc)
        accuracy = np.array(acc_history).mean()
        if verbose:
            print('>>> The accuracy on testing data is : {:.2%}'.format(accuracy))
        return accuracy

def plot(loss, acc_tr, acc_val):
    plt.subplot(121)
    plt.plot(loss)
    plt.title('loss curve')
    plt.xlabel('iteration')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.plot(acc_tr, label='acc_train')
    plt.plot(acc_val, label='acc_val')
    plt.legend()
    plt.title('training accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

#　prepare data train, validation, and test dataloader

batch_size = 100
split = 10000
index_val = np.random.choice(len(train_set), split, replace=False)
index_train = list(set(range(len(train_set))) - set(index_val))

train_sampler = torch.utils.data.SubsetRandomSampler(index_train)
val_sampler = torch.utils.data.SubsetRandomSampler(index_val)

train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 sampler=val_sampler)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1000,
                shuffle=False)

print('>>> batch amounts: train - {0}, val - {1}, test - {2}'.format(len(train_loader),len(val_loader),len(test_loader)))

use_saved_model = False

# create a LeNet module instance
inst = LeNet()
inst_3 = LeNet_3()
inst_leaky = LeNet_leaky()
inst_BN = LeNet_BN()

if use_saved_model:
    inst_BN.load_state_dict(torch.load('9917'))
    inst_BN.eval()
# # create a model
model = Net(inst_BN) 

SGD_update = dict(lr=1e-2, momentum=0.9, weight_decay=0.0)

loss_history, acc_train, acc_val = model.train(train_loader, val_loader, SGD_update, num_epoch=10)

toc = time.time()
print('>>> Total time cost --> {:.2f} minutes'.format(((toc - tic) / 60)))

model.predict(test_loader)

plot(loss_history, acc_train, acc_val)


