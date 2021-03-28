import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from .discriminator import Discriminator, SepDiscriminator
import os
if __name__ == '__main__':
    if not os.path.exists('/usr/xtmp/bbp13/cs590/particle_classifier/'):
        os.mkdir('/usr/xtmp/bbp13/cs590/particle_classifier/')
    EPOCHS = 100
    start_epoch = 0
    net = Discriminator()
    for i in range(EPOCHS):
        if i == 50:
            dec = 0.95
        print(datetime.datetime.now())
        # Switch to train mode
        net.train()
        print("Epoch %d:" %i)

        total_examples = 0
        correct_examples = 0

        train_loss = 0
        train_acc = 0

        # Train the training dataset for 1 epoch.
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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
            _, predicted = outputs.max(1)
            total_examples += predicted.size(0)
            correct_examples += predicted.eq(targets).sum().item()
            train_loss += loss
            global_step += 1

        avg_loss = train_loss / (batch_idx + 1)
        avg_acc = correct_examples / total_examples
        train_loss = avg_loss
        train_acc = avg_acc
        print("Training loss: %.4f, Training accuracy: %.4f" %(train_loss, train_acc))
        print(datetime.datetime.now())
        # Validate on the validation dataset
        print("Validation...")
        total_examples = 0
        correct_examples = 0

        net.eval()


        val_loss = 0
        val_acc = 0
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
                _, predicted = outputs.max(1)
                total_examples += predicted.size(0)
                correct_examples += predicted.eq(targets).sum().item()
                val_loss += loss

        avg_loss = val_loss / len(valloader)
        avg_acc = correct_examples / total_examples

        print("Validation loss: %.4f, Validation accuracy: %.4f" % (avg_loss, avg_acc))

        # Handle the learning rate scheduler.
        if i % DECAY_EPOCHS == 0 and i != 0:
            current_learning_rate = current_learning_rate * dec
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_learning_rate
            print("Current learning rate has decayed to %f" %current_learning_rate)

        # Save for checkpoint
        if avg_acc > best_val_acc:
            best_val_acc = avg_acc
            train_acc_of_best_val_acc = train_acc
            if not os.path.exists(CHECKPOINT_PATH):
                os.makedirs(CHECKPOINT_PATH)
            print("Saving ...")
            state = {'net': net.state_dict(),
                     'epoch': i,
                     'lr': current_learning_rate}
            torch.save(state, os.path.join(CHECKPOINT_PATH, filename))

    print("Optimization finished.")
    return [train_acc_of_best_val_acc, best_val_acc]