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

def changeDimension(x,num_channels):
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

class model_hyperparam(object):
    def __init__(self,learning_rate,batch_size,epochs,momentum,weight_decay,num_channels):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_channels = num_channels

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
        self.batchnorm = nn.BatchNorm2d(output_channel) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    
class Inception(nn.Module):
    def __init__(self, input_channel, output1, output3):
        super(Inception,self).__init__()
        self.branch1 = Conv(input_channel, output1, 1, 1,padding=0)
        self.branch3 = Conv(input_channel, output3, 3, 1,padding=1)

    def forward(self, x):
        b1 = self.branch1(x)  
        b3 = self.branch3(x) 
        # print(b1.shape,b3.shape)
        # Concatenate along the channel dimension
        return torch.cat([b1, b3], dim=1)

class Downsample(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Downsample,self).__init__()
        self.branch_conv = Conv(input_channel, output_channel, 3, 2,padding=1)
        self.branch_pool = nn.MaxPool2d(3, stride=2,padding=1)

    def forward(self, x):
        b_conv = self.branch_conv(x)  
        b_pool = self.branch_pool(x) 
        # print(b_conv.shape,b_pool.shape)
        # Concatenate along the channel dimension
        return torch.cat([b_conv, b_pool], dim=1)

class InceptionSmall(nn.Module):
    def __init__(self,input_channel):
        super(InceptionSmall, self).__init__()
        self.initial_conv = Conv(input_channel, 96, 3, 1,1)  

        # First Inception Block
        self.inception1 = Inception(96, 32, 32)
        self.inception2 = Inception(64, 32, 48)
        self.downsample1 = Downsample(80, 80)

        # Second Inception Block
        self.inception3 = Inception(160, 112, 48)
        self.inception4 = Inception(160, 96, 64)
        self.inception5 = Inception(160, 80, 80)
        self.inception6 = Inception(160, 48, 96)
        self.downsample2 = Downsample(144, 96)

        # Final Inception Block
        self.inception7 = Inception(240, 176, 160)
        self.inception8 = Inception(336, 176, 160)

        # Classification Head
        self.global_pool = nn.AvgPool2d(7)  # 7x7 kernel global pooling
        self.fc = nn.Linear(336, 10)  # 10-way classification

    def forward(self, x):
        x = self.initial_conv(x)
        # print('initial_conv done')
        x = self.inception1(x)
        # print('inception1 done')
        x = self.inception2(x)
        # print('inception2 done')

        x = self.downsample1(x)
        # print('downsample1 done')
        x = self.inception3(x)
        # print('inception3 done')

        x = self.inception4(x)
        # print('inception4 done')

        x = self.inception5(x)
        # print('inception5 done')

        x = self.inception6(x)
        # print('inception6 done')

        x = self.downsample2(x)
        # print('downsample2 done',x.shape)
        x = self.inception7(x)
        # print('inception7 done')

        x = self.inception8(x)
        # print('inception8 done')

        x = self.global_pool(x)
        # print('global_pool done')
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
def train_loop(dataloader, model, loss_fn, optimizer,device,batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    train_loss,correct = 0,0
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
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    train_loss/=num_batches
    print(f"Tranining accuracy: {(100*correct):>0.1f}%")
    return correct,train_loss


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
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_label",type=bool,default=False)
    parser.add_argument("--corrupt_percentage",type=int,default=0)

    args = parser.parse_known_args()[0]
    args_dict = vars(args)

    hyperparams = model_hyperparam(learning_rate=0.1,batch_size=128,epochs=5000,momentum=0.9,weight_decay=0.95,num_channels=3)
    
    directory = '/om2/user/shenwang/deeplearning/CIFAR/cifar-10-python/cifar-10-batches-py'
    model_folder = 'small_inception'
    data_prefix = 'data'
    test_prefix = 'test'

    training_files = glob.glob(directory+os.sep+data_prefix+'*')
    test_files = glob.glob(directory+os.sep+test_prefix+'*')

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

    training_raw_images = changeDimension(training_raw_images,hyperparams.num_channels)
    training_labels = changeDimension(training_labels,hyperparams.num_channels)
    test_raw_images = changeDimension(test_raw_images,hyperparams.num_channels)
    test_labels = changeDimension(test_labels,hyperparams.num_channels)

    training_images = preprocessing(training_raw_images)
    test_images = preprocessing(test_raw_images)

    for arg in args_dict:
        if arg == 'random_label' and args_dict[arg]:
            print('shuffle labels')
            training_labels = torch.randint(0, 10, training_labels.shape)
        if arg == 'corrupt_percentage':
            random_indices = torch.randperm(len(training_labels))[:int(len(training_labels)*args_dict[arg]/10)]
            print('Number of corrupt labels:',random_indices.shape)
            training_labels[random_indices] = torch.randint(0, 10, (len(random_indices),)) 
    
    plot_flags = ''
    if args.random_label:
        plot_flags+='random_labels'
        hyperparams.weight_decay = 1
    elif args.corrupt_percentage:
        plot_flags+='corrupt_labels_'+str(args.corrupt_percentage*10)
        hyperparams.weight_decay = 1

    else:
        plot_flags+='true_labels'
        
    plotdir = (
        plot_flags
        + "_"
        + time.strftime("%m-%d-%Y_%H-%M-%S")
    )
    
    plotdir = os.path.join(directory, model_folder, plotdir)
    args.plot_dir = plotdir
    if not (os.path.exists(plotdir)):
        os.makedirs(plotdir)

    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(device)


    data_train = CIFAR(torch.tensor(training_images,dtype=torch.float32),torch.tensor(training_labels,dtype=torch.long))
    data_test = CIFAR(torch.tensor(test_images,dtype=torch.float32),torch.tensor(test_labels,dtype=torch.long))

    train_dataloader = DataLoader(data_train, batch_size= hyperparams.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_dataloader = DataLoader(data_test, batch_size = hyperparams.batch_size,shuffle=True, num_workers=4,pin_memory=True)

    model = InceptionSmall(3).to(device)
    
    # model = InceptionSmall(3).to(device) # we do not specify ``weights``, i.e. create untrained model
    # model.load_state_dict(torch.load(directory + os.sep + 'model' + os.sep + 'model_weights0.pth', weights_only=True))

    optimizer = optim.SGD(model.parameters(), lr = hyperparams.learning_rate,momentum = hyperparams.momentum)
    loss_fn = nn.CrossEntropyLoss()
    print('weight decay is:',hyperparams.weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=hyperparams.weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    start_time = time.time()

    for t in range(hyperparams.epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_correct, train_loss = train_loop(train_dataloader, model, loss_fn, optimizer,device,hyperparams.batch_size)
        
        test_correct, test_loss = test_loop(test_dataloader, model, loss_fn,device)
        scheduler.step()
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f'elapsed time is:{elapsed_time} seconds')
        if t % 50 ==0:
            torch.save(model.state_dict(), plotdir + os.sep + 'model_weights'+str(t)+'.pth')
        if t % 10==0:
            print(f'saving running results')
        with open(plotdir + os.sep + 'file' + str(t) +'.pkl', 'wb') as file:
            pickle.dump([train_correct,train_loss,test_correct,test_loss], file)
    print("Done!")

if __name__ == '__main__':
    main()