import time
import torch
import torchvision
import torch.nn as nn
import os
from dataset_class import Custom_Data
from discriminator import Discriminator
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    CHECKPOINT_PATH = 'models/'
    EPOCHS = 10
    start_epoch = 0
    net = Discriminator(1, 32)
    criterion = nn.BCELoss()
    best_acc = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    if device =='cuda':
        print("Train on GPU...")
    else:
        print("Train on CPU...")
    
    

    optimizer = optim.SGD(net.parameters(), lr=.01)
   
    transform_train  = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),

    ])    
    
    train = Custom_Data('train3.csv', transform_train)

    val = Custom_Data('val.csv', transform_train) 
    trainloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
    valloader = torch.utils.data.DataLoader(val, batch_size=1, shuffle=True)
    
    for i in range(EPOCHS):

        # Switch to train mode
        net.train()
        print("Epoch %d:" %i)

        total_examples = 0
        correct_examples = 0

        train_loss = 0
        train_acc = 0


        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #print(inputs)
            #print(targets)
            
            # Your code: Copy inputs to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Your code: Zero the gradient of the optimizer (1 Line)
            optimizer.zero_grad()
            # Your code: Generate output (1 Line)
            outputs = net(inputs)
            # Your code: Compute loss (1 Line)
            loss = criterion(outputs, targets)
            # Your code: Now backward loss and compute gradient (1 Line)
            loss.backward()
            # Your code: apply gradient (1 Line)
            optimizer.step()
            # Calculate predicted labels
            predicted = outputs > .5
            #print(outputs, predicted, targets)
            #print('targets', targets.item())
            total_examples += 1 #predicted.size(0)
            correct_examples += predicted == targets
            train_loss += loss
           

        avg_loss = train_loss / (batch_idx + 1)
        avg_acc = float(correct_examples) / total_examples
        train_loss = avg_loss
        train_acc = avg_acc
        print("Training loss: %.4f, Training accuracy: %.4f" %(train_loss, train_acc))
        #print(datetime.datetime.now())
        '''
        if train_acc > best_acc:
            best_acc = train_acc
            print('Saving')
            state = {'net': net.state_dict(),
                     'epoch': i,
                     'lr': .01}
            torch.save(state, os.path.join(CHECKPOINT_PATH, f'model.pt'))
        '''
        #Validate on the validation dataset
        print("Validation...")
        total_examples = 0
        correct_examples = 0
       

        net.eval()

        
        val_loss = 0
        val_acc = 0
        best_val_acc = 0
        # Disable gradient during validation
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                # Copy inputs to device
                inputs = inputs.to(device)
                targets = targets.to(device)
                # Zero the gradient
                optimizer.zero_grad()
                # Generate output from the DNN.
                outputs = net(inputs)
                
                loss = criterion(outputs, targets)            
                # Calculate predicted labels
                predicted = outputs > .5 #outputs.max(1)
                print(outputs.item(), predicted.item(), targets.item())
                total_examples += 1 #predicted.size(0)
                
                correct_examples += predicted == targets #.eq(targets).sum().item()
                val_loss += loss

        avg_loss = val_loss / len(valloader)
        avg_acc = float(correct_examples) / total_examples

        print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))


        # Save for checkpoint
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            train_acc_of_best_val_acc = train_acc
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            print("Saving ...")
            state = {'net': net.state_dict(),
                     'epoch': i,
                     'lr': .01}
            torch.save(state, os.path.join(CHECKPOINT_PATH, f'model.pt'))

    print("Optimization finished.")
    #return [train_acc_of_best_val_acc, best_val_acc]
  