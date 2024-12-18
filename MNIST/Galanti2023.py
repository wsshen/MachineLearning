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

from keras.datasets import mnist
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import clip_grad_norm_

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# def changeDimension(x,num_channels):
#     assert isinstance(x,list),'x must be list type'
#     np_x = np.array(x)
#     # print(np_x.shape)
#     sp = np_x.shape
#     size_per_channel = sp[-1]/num_channels
#     len_per_side = int(np.sqrt(size_per_channel))
#     if len(sp) == 2:
#         new_array = np.reshape(np_x,(sp[0]*sp[1]))
#     if len(sp) == 3:
#         new_array = np.reshape(np_x,(sp[0]*sp[1],num_channels,len_per_side,len_per_side))
#         # sp_new = output.shape
#         # new_array = np.zeros((sp_new[0],sp_new[2],sp_new[3],sp_new[1]))
#         # for i in range(sp_new[0]):
#         #     for j in range(sp_new[1]):
#         #         new_array[i,:,:,j] = output[i,j,:,:]

#     return new_array

# def preprocessing(x,c_x=28,c_y=28,normalize=True,center_crop=True,whitening=True):
#     sp = x.shape
#     assert len(sp) == 4, 'The input shape must be number_of_frames * number_of_channels * len_of_image * len_of_image'
#     len_x = sp[2]
#     len_y = sp[3]
#     start_x = (len_x - c_x)//2
#     stop_x = start_x + c_x
#     start_y = (len_y-c_y)//2
#     stop_y = start_y + c_y
#     new_x = np.zeros((sp[0],sp[1],c_x,c_y))

#     for i in range(sp[0]):
#         for j in range(sp[1]):
#             if normalize:
#                 image = x[i,j,:,:]/255
#             else:
#                 image = x[i,j,:,:]
#             if center_crop:
#                 new_x[i,j,:,:] = image[start_x:stop_x,start_y:stop_y]
#             else:
#                 new_x[i,j,:,:] = image

#             if whitening:
#                 temp = image[start_x:stop_x,start_y:stop_y]
#                 mean = np.mean(temp)
#                 std = np.std(temp)
#                 std_mod = max(std,1/np.sqrt(np.size(temp)))
#                 new_x[i,j,:,:] = (temp - mean)/std_mod
#     return new_x

class model_hyperparam(object):
    def __init__(self,learning_rate,batch_size,epochs,momentum,decay_factor,num_channels):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.decay_factor = decay_factor
        self.num_channels = num_channels

class MNIST(Dataset):
    def __init__(self,data,label):
        super(MNIST,self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx,:,:], self.label[idx]
    
class Conv(nn.Module):
    def __init__(self,input_channel,output_channel,kernel_size,stride,padding):
        super(Conv, self).__init__()
        self.conv = weight_norm(nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding,bias=False) )      
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    
    
class CNN(nn.Module):
    def __init__(self,input_channel):
        super(CNN, self).__init__()
        self.conv1 = Conv(input_channel,200,2,1,0)
        self.conv2 = Conv(200,200,2,1,0)
        self.conv3 = Conv(200,200,2,1,0)

        self.fc = nn.Linear(125000,10,bias=False)

    def forward(self, x):
        x = self.conv1(x)
        # print('conv1 done',x.shape)
        x = self.conv2(x)
        # print('conv2 done',x.shape)
        x = self.conv3(x)
        # print('conv3 done',x.shape)
        x = torch.flatten(x, 1)
        # print('after flattening',x.shape)
        x = self.fc(x)
        # print('fc done')


        return x
    
def train_loop(dataloader, model, loss_fn, optimizer,device,batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    gradient_clip = 1.0
    train_loss,correct = 0,0
    num_batches = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X,y = X.to(device),y.to(device)
        X = X.view(-1, 1, 28, 28)
        pred = model(X)
        if isinstance(loss_fn,nn.modules.loss.MSELoss):
            y_new = F.one_hot(y,num_classes=10).float()
        else:
            y_new = y
        loss = loss_fn(pred, y_new)

        loss.backward()
        clip_grad_norm_(model.parameters(),gradient_clip)
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
            X = X.view(-1, 1, 28, 28)

            pred = model(X)

            if isinstance(loss_fn,nn.modules.loss.MSELoss):
                y_new = F.one_hot(y,num_classes=10).float()
            else:
                y_new = y

            test_loss += loss_fn(pred, y_new).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct,test_loss
    
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_label",type=bool,default=False)
    parser.add_argument("--corrupt_percentage",type=float,default=0.0)

    args = parser.parse_known_args()[0]
    args_dict = vars(args)
    print(args,args_dict)

    hyperparams = model_hyperparam(learning_rate=1e-2,batch_size=32,epochs=500,momentum=0.9,decay_factor=0.1,num_channels=1)

    directory = '/home/watson/Documents/CIFAR/cifar-10-python/cifar-10-batches-py'
    model_folder = 'Galanti2023'
    # data_prefix = 'data'
    # test_prefix = 'test'

    # training_files = glob.glob(directory+os.sep+data_prefix+'*')
    # test_files = glob.glob(directory+os.sep+test_prefix+'*')
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()


    # training_raw_images = []
    # training_labels = []
    # test_raw_images = []
    # test_labels = []

    # for file in training_files:
    #     batch_dict = unpickle(file)
    #     training_raw_images.append(batch_dict[b'data'])
    #     training_labels.append(batch_dict[b'labels'])
    # for file in test_files:
    #     batch_dict = unpickle(file)
    #     test_raw_images.append(batch_dict[b'data'])
    #     test_labels.append(batch_dict[b'labels'])

    # training_raw_images = changeDimension(training_raw_images,hyperparams.num_channels)
    # training_labels = changeDimension(training_labels,hyperparams.num_channels)
    # test_raw_images = changeDimension(test_raw_images,hyperparams.num_channels)
    # test_labels = changeDimension(test_labels,hyperparams.num_channels)

    # training_images = preprocessing(training_raw_images)
    # test_images = preprocessing(test_raw_images)

    for arg in args_dict:
        if arg == 'random_label' and args_dict[arg]:
            print('shuffle labels')
            training_labels = torch.randint(0, 10, training_labels.shape) 
        if arg == 'corrupt_percentage':
            random_indices = torch.randint(0, len(training_labels), (len(training_labels*args_dict[arg]),))
            training_labels[random_indices] = torch.randint(0, 10, (len(random_indices),)) 

    plot_flags = ''
    if args.random_label:
        plot_flags+='random_labels'
    elif args.corrupt_percentage:
        plot_flags+='corrupt_labels_'+str(args.corrupt_percentage)
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
    # device = torch.device("cpu")
    print(device)

    

    data_train = MNIST(torch.tensor(training_images,dtype=torch.float32),torch.tensor(training_labels,dtype=torch.long))
    data_test = MNIST(torch.tensor(test_images,dtype=torch.float32),torch.tensor(test_labels,dtype=torch.long))

    train_dataloader = DataLoader(data_train, batch_size= hyperparams.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_dataloader = DataLoader(data_test, batch_size=hyperparams.batch_size,shuffle=True, num_workers=4,pin_memory=True)

    model = CNN(1).to(device)
    
    # model = InceptionSmall(3).to(device) # we do not specify ``weights``, i.e. create untrained model
    # model.load_state_dict(torch.load(directory + os.sep + 'model' + os.sep + 'model_weights0.pth', weights_only=True))


    optimizer = optim.SGD(model.parameters(), lr=hyperparams.learning_rate,momentum=hyperparams.momentum,weight_decay=3e-3) #
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,100,300], gamma=hyperparams.decay_factor)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")

    start_time = time.time()

    for t in range(hyperparams.epochs):
        print(f"Epoch {t+1}\n-------------------------------")

        train_correct,train_loss = train_loop(train_dataloader, model, loss_fn, optimizer,device,hyperparams.batch_size)
        
        test_correct, test_loss = test_loop(test_dataloader, model, loss_fn,device)
        scheduler.step()
        current_time = time.time()
        elapsed_time = current_time - start_time
        print(f'elapsed time is:{elapsed_time} seconds')
        if t % 50 ==0:
            torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, plotdir + os.sep + 'model_weights'+str(t)+'.pth')
        
        if t % 10==0:
            print(f'saving running results')
        
            with open(plotdir + os.sep + 'file' + str(t) +'.pkl', 'wb') as file:
                pickle.dump([train_correct,train_loss,test_correct,test_loss], file)
    print("Done!")

if __name__ == '__main__':
    main()