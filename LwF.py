import numpy as np
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

# One hot
def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class LwF():
    def __init__(self,memory=2000,device='cuda',params=None, train_batches, test_batches):
        self.memory = memory
        self.device = device
        self.params = params
        self.train_batches = train_batches
        self.test_batches = test_batches
        self.old_model = None
        self.TASK_SIZE = self.params['TASK_SIZE']
        
    # Method to get dataloaders step by step to not break the memory #
    def dataloaders(self, train, test):
        train_loader = DataLoader(dataset=train,
                                  shuffle=True,
                                  batch_size=self.batchsize)

        test_loader = DataLoader(dataset=test,
                                 shuffle=True,
                                 batch_size=self.batchsize)

        return train_loader, test_loader
    
    # Define Distillation Loss #
    def D_loss(net, images, target):
        output = net(images)
        target = get_one_hot(target, net.num_classes)
        output = output.to(DEVICE)
        target = target.to(DEVICE)

        if net.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            old_target = torch.sigmoid(net.old_model(images))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)
    
    # Warm-up step #
    def beforeTrain(net):
        net.eval()
            if net.num_classes > self.TASK_SIZE:
                net.Incremental_learning(net.num_classes) # resnet32 Cifar
        net.train()        
        net.to(DEVICE)
    
    # Method to compute val_accuracy #
    def test(net, testloader,test_batch):
        net.eval()
        running_corrects = 0
        for images, labels, indexes in (testloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
        
            outputs = net(images)
            _, preds = torch.max(outputs.data, 1)
            running_corrects += torch.sum(preds == labels.data).data.item()
    
        accuracy = running_corrects / len(test_batch)
        return accuracy
    

    # Train step #
    def train(net, batch, test_batch, trainBatchLoader , testBatchLoader):
        print('\n ### Updating Representation ###')
        EPOCHS = self.params['EPOCHS']
        BATCH_SIZE = self.params['BATCH_SIZE']
        LR = self.params['LR']
        MOMENTUM = self.params['MOMENTUM']
        WEIGHT_DECAY = self.params['WEIGHT_DECAY']   
        
        optimizer = optim.SGD(net.parameters(), lr = LR, momentum = MOMENTUM, nesterov=True, weight_decay = WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=(STEP_SIZE-1), factor=GAMMA)
        
        best_accuracy = 0.0
        best_epoch = 0

        net = net.to(DEVICE)
   
        for epoch in range(NUM_EPOCHS):
            print('Starting epoch {}/{}, LR = {}, time: {} minutes'.format(epoch+1, NUM_EPOCHS, optimizer.param_groups[0]['lr'], round((time.time()-t0)/60,2)))
            net.train() # Sets module in training mode
       
            # Initialize variables
            running_class_loss = 0.0
            running_dist_loss = 0.0

            running_loss = 0.0
            running_corrects = 0
       
            for images,labels,indexes in trainBatchLoader:
            
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero-ing the gradients
                optimizer.zero_grad() 

                # Compute loss
                loss = D_loss(net, images, target)
                print('LOSS: {}'.format(loss))
               
                # Update running corrects
                _, preds = torch.max(outputs.data, 1)
                running_corrects += torch.sum(preds == labels.data).data.item()

                # Compute gradients for each layer and update weights
                tot_loss.backward()  
                optimizer.step() # Update weights based on accumulated gradients

            train_accuracy = running_corrects / len(batch)

        # Best validation model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = net
            best_epoch = epoch
            print(' ### Best_model updated\n')

        scheduler.step(LR)

    print(f'{round((time.time()-t0)/60,2)} minutes spent on training')
    print(f'Best validation accuracy {best_accuracy} reached at epoch {(best_epoch+1)}')
    
    # After train: update old_model and increment number of classes #
    def afterTrain(net):    
        # ADD number of tasks
        net.num_classes += self.TASK_SIZE
        # Save old net 
        net.old_model = deepcopy(net)
        net.old_model.to(DEVICE)
        net.old_model.eval()
        
    def run(net):
        for trainB, testB in zip(self.train_batches, self.test_batches):
            trainLoaders, testLoaders = dataloaders(self, trainB, testB)
            beforeTrain(net)
            train(net, trainB, testB, trainLoaders, testLoaders)
            afterTrain(net)