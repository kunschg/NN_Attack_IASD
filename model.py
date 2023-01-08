#!/usr/bin/env python3

"""
GBH Module model.py
Defines the models that are designed to be robust against adversarial attacks.
"""
import os
import argparse
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import sys
from tqdm import tqdm
from CIFARLoader import CIFAR_Loader
from settings import device, batch_size, git_dir, PGD_EPS,PGD_STEPSIZE,PGD_NB_ITERS,PGD_TRAIN_ALPHA
from pathlib import Path
from attacks import pgd_attack,batch_pgd_attack
from test_project import test_PGD_attacked_accuracy

# ========== ARCHITECTURES ==========

'''Basic neural network architecture (from pytorch doc).'''
class Net(nn.Module):

    model_file=Path("models","PGD_20_epochs2.pth").__str__()
    '''This file will be loaded to test your model. Use --model-file to load/store a different model.'''
    print(f"{model_file=}")
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

    def save(self, model_file):
        '''Helper function, use it to save the model weights after training.'''
        torch.save(self.state_dict(), model_file)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file, map_location=torch.device(device)))

        
    def load_for_testing(self, project_dir='./'):
        '''This function will be called automatically before testing your
           project, and will load the model weights from the file
           specify in Net.model_file.
           
           You must not change the prototype of this function. You may
           add extra code in its body if you feel it is necessary, but
           beware that paths of files used in this function should be
           refered relative to the root of your project directory.
        '''        
        self.load(os.path.join(project_dir, Net.model_file))


'''VAE architecture'''
class VAE(nn.Module):
    def __init__(self, channels, latent_dim):
        super(VAE, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim

        # encoder
        self.pool = nn.MaxPool2d(2,2)
        self.conv_enc1, self.conv_enc2 = [], []
        self.bnorm_enc = []
        for i in range(1,len(channels)):
          self.conv_enc1.append(nn.Conv2d(channels[i-1], channels[i], kernel_size=4, padding='same'))
          self.conv_enc2.append(nn.Conv2d(channels[i], channels[i], kernel_size=4, padding='same'))
          self.bnorm_enc.append(nn.BatchNorm2d(channels[i]))
        self.conv_enc1 = nn.ModuleList(self.conv_enc1)
        self.conv_enc2 = nn.ModuleList(self.conv_enc2)
        self.bnorm_enc = nn.ModuleList(self.bnorm_enc)
        self.fc1 = nn.Linear(channels[-1] * int(32 / (2**(len(channels)-2))), latent_dim)

        # decoder
        self.fc2 = nn.Linear(latent_dim, channels[-1] * int(32 / (2**(len(channels)-2))))
        self.conv_dec1, self.conv_dec2 = [], [] 
        self.bnorm_dec = []
        for i in range(len(channels)-1, 0, -1):
          self.conv_dec1.append(nn.ConvTranspose2d(channels[i], channels[i-1], stride = 2, kernel_size=4, padding=1))
          self.conv_dec2.append(nn.Conv2d(channels[i-1], channels[i-1], kernel_size=4, padding='same'))
          self.bnorm_dec.append(nn.BatchNorm2d(channels[i-1]))
        self.conv_dec1 = nn.ModuleList(self.conv_dec1)
        self.conv_dec2 = nn.ModuleList(self.conv_dec2)
        self.bnorm_dec = nn.ModuleList(self.bnorm_dec)

    def encode(self, x):
        for i in range(len(channels)-1):
            #x = self.pool(F.relu(self.conv_enc2[i](F.relu(self.conv_enc1[i](x)))))
            x = self.pool(F.relu(self.bnorm_enc[i](self.conv_enc2[i](F.relu(self.bnorm_enc[i](self.conv_enc1[i](x)))))))
        x = x.view(-1, channels[-1] * int(32 / (2**(len(channels)-2))))
        mu = self.fc1(x)
        log_var = self.fc1(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc2(z)
        x = x.view(-1, channels[-1], int(32 / (2**(len(channels)-1))), int(32 / (2**(len(channels)-1))))
        for i in range(len(channels)-2):
            #x = F.relu(self.conv_dec2[i](F.relu(self.conv_dec1[i](x))))
            x = F.relu(self.bnorm_dec[i](self.conv_dec2[i](F.relu(self.bnorm_dec[i](self.conv_dec1[i](x))))))
        x = self.conv_dec1[-1](x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z_reparametrized = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, log_var

'''
VAE classifier: takes images as input, then makes them through a VAE before classifying the reconstructed images
Inputs:
- vae_model = VAE(channels, latent_dim)
- clf_model = Net()
- vae_weights and clf_weights are .pth files 
'''
class VAEClassifier(nn.Module):
    def __init__(self, vae_model, vae_weights, clf_model, clf_weights):
        super(VAEClassifier, self).__init__()
        self.vae = vae_model
        self.vae.load_state_dict(torch.load(vae_weights))
        self.clf = clf_model
        self.clf.load_state_dict(torch.load(clf_weights))
      
    def forward(self, x):
        reconstructed_image, _, _ = self.vae(x)
        pred = self.clf(reconstructed_image)
        return pred

# ========== TRAINING FUNCTIONS ==========

def train_model(net, train_loader,validation_loader,
                pth_filename, num_epochs,patience=10,lr=1e-3):
    '''
    Training the model with natural images
    args:
        net: the model itself
        train_loader: train dataset loader
        validation_loader: to evaluate the model on unseen data
        pth_filename: checkpoint where to save the model
        num_epochs: number of total epochs
        patience: stop the training if evaluation metric doesn't increase after
            patience epochs
        lr: learning rate for adam optimizer
    returns:

    '''
    print("Starting training")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #evaluation loader to compare the metrics with unseen images
    # images_eval,labels_eval=validation_loader
    # images_eval, labels_eval = images_eval.to(device), labels_eval.to(device)
    # to get save the checkpoint model if the performance increases
    max_acc_eval = 0
    train_losses = []
    eval_losses = []
    counter = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        for images, labels in tqdm(train_loader,leave=False,desc="Training"):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = images.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        #evaluating the model
        with torch.no_grad():
            #accumulated evaluation and loss on the evaluation set
            acc_eval=0
            cpt_batches=0
            acc_loss=0
    
            for images_eval,labels_eval in tqdm(validation_loader,leave=False, desc="Validation"):
                images_eval, labels_eval = images_eval.to(device), labels_eval.to(device)
                out_eval = net(images_eval)
                acc_loss+= criterion(out_eval, labels_eval)    
                _, labels_pred = torch.max(out_eval, 1)
                labels_pred = labels_pred.to(device)
                acc_eval+= np.mean([((labels_pred[i] == labels_eval[i])*1).item() for i in range(len(labels_eval))])
                cpt_batches+=1
            loss_eval=acc_loss/cpt_batches
            acc_eval=acc_eval/cpt_batches
            print(f'Training loss: {loss.item():.4f}, Evaluation loss: {loss_eval.item():.4f},\
                    Evaluation accuracy: {round(acc_eval*100,1)}%')
        if acc_eval <= max_acc_eval:
            counter += 1
        else:
            max_acc_eval = acc_eval
            net.save(pth_filename)
            counter = 0
    
        train_losses.append(loss.cpu().detach())
        eval_losses.append(loss_eval.cpu().detach())

        if counter == patience:
            break
    print(f'Finished training model. Saved best weights in {pth_filename}')
    return train_losses,eval_losses


def train_model_robust_PGD(net, train_loader,validation_loader,
                pth_filename, num_epochs,patience=10,lr=1e-3,
                    iters=PGD_NB_ITERS, epsilon=PGD_EPS, step_size=PGD_STEPSIZE,targeted=False):
    '''
    Training the model with natural images
    args:
        net: the model itself
        train_loader: train dataset loader
        validation_loader: to evaluate the model on unseen data
        pth_filename: checkpoint where to save the model
        num_epochs: number of total epochs
        patience: stop the training if evaluation metric doesn't increase after
            patience epochs
        lr: learning rate for adam optimizer
    returns:
    '''

    print("Starting robust training for PGD")
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #evaluation loader to compare the metrics with unseen images
    # images_eval,labels_eval=validation_loader
    # images_eval, labels_eval = images_eval.to(device), labels_eval.to(device)
    # to get save the checkpoint model if the performance increases
    max_acc_eval = 0
    train_losses = []
    eval_losses = []
    counter = 0

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        for images, labels in tqdm(train_loader,leave=False,desc="Training PGD mode"):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = images.to(device), labels.to(device)
            # zero the parameter gradients
            # computing the attacked images to compute a global loss
            attacked_images,*_ = batch_pgd_attack(model=net,victim_images_labels=(images,labels),
                    device=device, iters=iters, eps=epsilon, step_size=step_size,
                        targets=None,verbose=False)
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            if not targeted:
                targets=labels
            loss = adversarial_loss(net=net,natural_images=inputs,alpha=PGD_TRAIN_ALPHA,
                            attacked_images=attacked_images,targets=targets,criterion=criterion)
            loss.backward()
            optimizer.step()

        #evaluating the model
        with torch.no_grad():
            #accumulated evaluation and loss on the evaluation set
            acc_eval=0
            cpt_batches=0
            acc_loss=0
            for images_eval,labels_eval in tqdm(validation_loader,leave=False, desc="Validation"):
                images_eval, labels_eval = images_eval.to(device), labels_eval.to(device)
                out_eval = net(images_eval)
                acc_loss+= criterion(out_eval, labels_eval)    
                _, labels_pred = torch.max(out_eval, 1)
                labels_pred = labels_pred.to(device)
                acc_eval+= np.mean([((labels_pred[i] == labels_eval[i])*1).item() for i in range(len(labels_eval))])
                cpt_batches+=1
            loss_eval=acc_loss/cpt_batches
            acc_eval=acc_eval/cpt_batches
            acc_pgd = test_PGD_attacked_accuracy(net, test_loader=validation_loader,
                                                device=device,iters=iters,eps=epsilon,step_size=step_size,num_samples=1)
            print(f'Training loss: {loss.item():.4f}, Evaluation loss: {loss_eval.item():.4f},\
                    Evaluation accuracy: {round(acc_eval*100,1)}%, PGD attack accuracy: {acc_pgd}')
      
        if acc_eval <= max_acc_eval:
            counter += 1
        else:
            max_acc_eval = acc_eval
            net.save(pth_filename)
            counter = 0
    
        train_losses.append(loss.cpu().detach())
        eval_losses.append(loss_eval.cpu().detach())

        if counter == patience:
            break
    print(f'Finished training model. Saved best weights in {pth_filename}')
    return train_losses,eval_losses

def train_vae(net, train_loader, validation_loader, pth_filename, num_epochs, patience=10, lr=1e-3):
    '''
    Function to train the VAE
    '''
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss(reduction='sum')
    inputs_eval = next(iter(validation_loader))[0].to(device)
    min_loss_eval = np.inf
    counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        total_recon_loss = 0
        total_kl_loss = 0
        total_loss = 0
        
        for inputs, _ in tqdm(train_loader):
            inputs = inputs.to(device)
            
            # Pass the inputs through the VAE
            outputs, mu, log_var = net(inputs)
            
            # Calculate the reconstruction loss
            recon_loss = loss_fn(outputs, inputs)
            
            # Calculate the KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # Calculate the total loss
            loss = recon_loss + kl_loss
            
            # Backpropagate the loss and update the model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Increment total losses
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()

        # Evaluation
        with torch.no_grad():
            outputs_eval, mu_eval, log_var_eval = net(inputs_eval)
            recon_loss_eval = loss_fn(outputs_eval, inputs_eval)
            kl_loss_eval = -0.5 * torch.sum(1 + log_var_eval - mu_eval.pow(2) - log_var_eval.exp())
            loss_eval = recon_loss_eval + kl_loss_eval

        # Patience
        if loss_eval >= min_loss_eval:
            counter += 1
        else:
            min_loss_eval = loss_eval
            torch.save(net.state_dict(), pth_filename)
            counter = 0

        print(f'Training: Average loss = {total_loss:.4f}, Reconstruction term = {total_recon_loss:.4f}, KL term = {total_kl_loss:.4f}')
        print(f'Evaluation: Average loss = {loss_eval:.4f}, Reconstruction term = {recon_loss_eval:.4f}, KL term = {kl_loss_eval:.4f}')

        if counter >= patience:
            break


def adversarial_loss(net, alpha, natural_images, attacked_images, targets, criterion):
    """
    Custom loss for the adversarial training outputing alpha*natural_loss + (1-alpha)*attaqued_loss
    args:
        net: model on which training occurs
        alpha: hyperparam tuning which loss gets the most importance
        natural_images: plain images that haven't been attaqued
        attacked_images: images on which was made the PGD/FGSM attack
        targets: labels for training the natural images and the attacked on untargetted
        criterion: loss function (mainly  nn.NLLLoss())
    """
    natural_outputs = net(natural_images)
    attacked_outputs = net(attacked_images)
    loss_natural = criterion(natural_outputs,targets)          
    loss_attacked = criterion(attacked_outputs,targets)
    return alpha*loss_natural + (1-alpha)*loss_attacked

def test_natural(net, test_loader):
    '''Basic testing function of the net model'''

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(test_loader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

def main():
    """
    Trains the model if asked for a evalutes its performances
    """
    #### Parse command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=Net.model_file,
                        help="Name of the file used to load or to sore the model weights."\
                        "If the file exists, the weights will be load from it."\
                        "If the file doesn't exists, or if --force-train is set, training will be performed, "\
                        "and the model weights will be stored in this file."\
                        "Warning: "+Net.model_file+" will be used for testing (see load_for_testing()).")
    parser.add_argument('-f', '--force-train', action="store_true",
                        help="Force training even if model file already exists"\
                             "Warning: previous model file will be erased!).")
    parser.add_argument('-e', '--num-epochs', type=int, default=10,
                        help="Set the number of epochs during training")
    parser.add_argument("--PGD-mode","--PGD_mode",action="store_true",help="Running the training in the robust to"\
                                                                'PGD mode.')
    args = parser.parse_args()

    #### Create model and move it to whatever device is available (gpu/cpu)
    net = Net()
    net.to(device)
    print(f"Using device: {device}")
    loader=CIFAR_Loader(batch_size=batch_size,train_eval_frac=0.9,root_folder=Path(git_dir,"data"))
    train_loader, validation_loader = loader.get_train_eval_loaders()
    test_loader = loader.get_test_loader()
    # a,b = next(iter(train_loader))
    # print("THERE IS ",a.shape)
    # a,b = next(iter(validation_loader))
    # print("THERE IS ",a.shape)
    # a,b = next(iter(test_loader))
    # print("THERE IS ",a.shape)
    #### Model training (if necessary)
    if not os.path.exists(args.model_file) or args.force_train:
        msg="using PGD mode." if args.PGD_mode else ""
        print(f"Training model {msg}")
        if args.PGD_mode:
            train_model_robust_PGD(net=net, train_loader=train_loader,validation_loader=validation_loader,
                                pth_filename= args.model_file, num_epochs= args.num_epochs)
        else:
            train_model(net=net, train_loader=train_loader,validation_loader=validation_loader,
                                pth_filename= args.model_file, num_epochs= args.num_epochs)
        print("Model save to '{}'.".format(args.model_file))

    #### Model testing
    print(f"Testing the model from {args.model_file}")

    # Note: You should not change the transform applied to the
    # validation dataset since, it will be the only transform used
    # during final testing.

    net.load(args.model_file)

    nat_acc = test_natural(net, validation_loader)


    print("Model natural accuracy (valid): {}".format(nat_acc))
    if args.PGD_mode:
        pgd_accuracy = test_PGD_attacked_accuracy(net, test_loader= validation_loader,device=device)
        print("Model accuracy on PGD attacked images (valid): {}".format(pgd_accuracy))

        
    if args.model_file != Net.model_file:
        print(f"Warning: {args.model_file} is not the default model file, "\
              "it will not be the one used for testing your project. "\
              "If this is your best model, "\
              f"you should rename/link {args.model_file} to {Net.model_file}.")


if __name__ == "__main__":
    main()