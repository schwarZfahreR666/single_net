# coding: utf-8




import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
import initialize
from tensorboardX import SummaryWriter
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mfnet_3d import MFNET_3D
from new_dataset import Image_Dataset






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

N_EPOCHS = 100  # Number of epochs for training
BATCH_SIZE=16
LR = 5e-2 # Learning rate
PRE_TRAINED=True
Model_path='models/MFNet3D_Kinetics-400_72.8.pth'
save_dir_root = os.path.join(os.path.abspath('.'))
save_dir_models=os.path.join(save_dir_root,'models')
if not os.path.exists(save_dir_models):
    os.makedirs(save_dir_models)
save_dir_logs=os.path.join(save_dir_root,'logs')
if not os.path.exists(save_dir_logs):
    os.makedirs(save_dir_logs)




model = MFNET_3D(num_classes=174,batch_size=BATCH_SIZE)
train_params = model.parameters()
criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
optimizer = optim.SGD(train_params, lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40,gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

# if torch.cuda.device_count() > 1:
#   model = nn.DataParallel(model)
model.to(device)
criterion.to(device)
if PRE_TRAINED:
	initialize.init_from_dict(model,Model_path,device,Flags=False)
	print('loaded pretrained model')
writer = SummaryWriter(log_dir=save_dir_logs)
train_dataloader = DataLoader(Image_Dataset('train_num2label.txt'), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_dataloader   = DataLoader(Image_Dataset('val_num2label.txt'), batch_size=BATCH_SIZE, num_workers=4)
test_dataloader  = DataLoader(Image_Dataset('test_num2label.txt'), batch_size=BATCH_SIZE, num_workers=4)

trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
test_size = len(test_dataloader.dataset)





for epoch in range(0, N_EPOCHS):
   
    print('Epoch:',epoch)
    for phase in ['train', 'val']:
        print(phase,':')
        
        running_loss = 0.0
        running_corrects = 0.0
        if phase == 'train':
            # scheduler.step() is to be called once every epoch during training
            scheduler.step()
            model.train()
        else:
            model.eval()
            
            
        for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                        
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / trainval_sizes[phase]
        epoch_acc = float(running_corrects.double() / trainval_sizes[phase])
        print('Loss:',epoch_loss)
        print('Acc:',epoch_acc)
        
        if phase == 'train':
            writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
        else:
            writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
        
    
   
    if epoch % 10 == 0:
        torch.save(model.state_dict(),os.path.join(save_dir_models,'trained_model.pkl'))
        print('model saved')
writer.close()

