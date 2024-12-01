import os
import glob
import numpy as np
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def changeDimension(x):
    assert isinstance(x,list),'x must be list type'
    np_x = np.array(x)
    # print(np_x.shape)
    sp = np_x.shape
    size_per_channel = sp[-1]/num_channels
    len_per_side = int(np.sqrt(size_per_channel))
    if len(sp) == 2:
        new_array = np.reshape(np_x,(sp[0]*sp[1]))
    if len(sp) == 3:
        new_array = np.reshape(np_x,(sp[0]*sp[1],num_channels,len_per_side,len_per_side))
        # sp_new = output.shape
        # new_array = np.zeros((sp_new[0],sp_new[2],sp_new[3],sp_new[1]))
        # for i in range(sp_new[0]):
        #     for j in range(sp_new[1]):
        #         new_array[i,:,:,j] = output[i,j,:,:]

    return new_array

def preprocessing(x,c_x=28,c_y=28,normalize=True,center_crop=True,whitening=True):
    sp = x.shape
    assert len(sp) == 4, 'The input shape must be number_of_frames * number_of_channels * len_of_image * len_of_image'
    len_x = sp[2]
    len_y = sp[3]
    start_x = (len_x - c_x)//2
    stop_x = start_x + c_x
    start_y = (len_y-c_y)//2
    stop_y = start_y + c_y
    new_x = np.zeros((sp[0],sp[1],c_x,c_y))

    for i in range(sp[0]):
        for j in range(sp[1]):
            if normalize:
                image = x[i,j,:,:]/255
            else:
                image = x[i,j,:,:]
            if center_crop:
                new_x[i,j,:,:] = image[start_x:stop_x,start_y:stop_y]
            else:
                new_x[i,j,:,:] = image

            if whitening:
                temp = image[start_x:stop_x,start_y:stop_y]
                mean = np.mean(temp)
                std = np.std(temp)
                std_mod = max(std,1/np.sqrt(np.size(temp)))
                new_x[i,j,:,:] = (temp - mean)/std_mod
    return new_x

class CIFAR(Dataset):
    def __init__(self,data,label):
        super(CIFAR,self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx,:,:,:], self.label[idx]
    
class Conv(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size,stride,padding):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.max = nn.MaxPool2d(3, stride=2,padding=1)
        self.localnorm = nn.LocalResponseNorm(5,alpha=1e-4,beta=0.75,k=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.max(x)
        x = self.localnorm(x)
        x = self.relu(x)
        return x
    
class fullconnect(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(fullconnect, self).__init__()
        self.fc = nn.Linear(input_channel,output_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x
    
class AlexnetSmall(nn.Module):
    def __init__(self,input_channel):
        super(AlexnetSmall, self).__init__()
        self.conv1 = Conv(input_channel,160,5,3,1)
        self.conv2 = Conv(160,256,5,3,0)
        self.fc1 = fullconnect(256,384)
        self.fc2 = fullconnect(384,192)
        self.fc3 = nn.Linear(192, 10)  # 10-way classification

    def forward(self, x):
        x = self.conv1(x)
        # print('conv1 done',x.shape)
        x = self.conv2(x)
        # print('conv2 done',x.shape)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        # print('fc1 done')
        x = self.fc2(x)
        # print('fc2 done')
        x = self.fc3(x)
        # print('fc3 done')
        return x
    
def train_loop(dataloader, model, loss_fn, optimizer,device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss = 0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # print(batch,X.shape,y.shape)
        X,y = X.to(device),y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss/=num_batches
    return train_loss


def test_loop(dataloader, model, loss_fn,device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X,y = X.to(device),y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct,test_loss
    

if __name__ == '__main__':
    directory = '/Users/shenwang/Documents/CIFAR/cifar-10-python/cifar-10-batches-py'
    data_prefix = 'data'
    test_prefix = 'test'
    num_channels = 3

    training_files = glob.glob(directory+os.sep+data_prefix+'*')
    test_files = glob.glob(directory+os.sep+test_prefix+'*')

    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")

    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    print(device)

    training_raw_images = []
    training_labels = []

    test_raw_images = []
    test_labels = []

    for file in training_files:
        batch_dict = unpickle(file)
        training_raw_images.append(batch_dict[b'data'])
        training_labels.append(batch_dict[b'labels'])
    for file in test_files:
        batch_dict = unpickle(file)
        test_raw_images.append(batch_dict[b'data'])
        test_labels.append(batch_dict[b'labels'])

    training_raw_images = changeDimension(training_raw_images)
    training_labels = changeDimension(training_labels)
    test_raw_images = changeDimension(test_raw_images)
    test_labels = changeDimension(test_labels)

    training_images = preprocessing(training_raw_images)
    test_images = preprocessing(test_raw_images)

    learning_rate = 0.01
    batch_size = 128
    epochs = 5000
    momentum = 0.9
    weight_decay = 0.95

    data_train = CIFAR(torch.tensor(training_images,dtype=torch.float32),torch.tensor(training_labels,dtype=torch.long))
    data_test = CIFAR(torch.tensor(test_images,dtype=torch.float32),torch.tensor(test_labels,dtype=torch.long))

    train_dataloader = DataLoader(data_train, batch_size= batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_dataloader = DataLoader(data_test, batch_size=batch_size,shuffle=True, num_workers=4,pin_memory=True)

    model = AlexnetSmall(3).to(device)
    
    # model = InceptionSmall(3).to(device) # we do not specify ``weights``, i.e. create untrained model
    # model.load_state_dict(torch.load(directory + os.sep + 'model' + os.sep + 'model_weights0.pth', weights_only=True))


    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    start_time = time.time()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer,device)
        
        test_correct, test_loss = test_loop(test_dataloader, model, loss_fn,device)
        scheduler.step()
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f'elapsed time is:{elapsed_time} seconds')
        if t % 50 ==0:
            torch.save(model.state_dict(), directory + os.sep + 'model_alexnet' + os.sep + 'model_weights'+str(t)+'.pth')
        if t % 10==0:
            print(f'saving running results')
        with open(directory + os.sep + 'model_alexnet' + os.sep + 'file' + str(t) +'.pkl', 'wb') as file:
            pickle.dump([train_loss,test_correct,test_loss], file)
    print("Done!")