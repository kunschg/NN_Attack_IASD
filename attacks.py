"""
    This module defines the attacks that are devolopped in the project (FGSM and PGD)
    along with some helper functions in order to easily compute the gradients wrt to 
    the input image for example
"""
import numpy as np 
import torch.nn as nn
import torch
from settings import PGD_EPS, PGD_NB_ITERS,PGD_STEPSIZE


def compute_gradient(model,loss_function,input_images,target):
    """
    computes the gradient of a loss_function given its input_images
    args:
        model: the pytorch model on which to perform the attack
        loss_function: the loss function on which to compute the gradient
        input_images: the image to attack
        target: the label (either towards which to go for targeted attacks
            or from which to maximise the step for untargeted attacks)
    """
    input_images.requires_grad=True
    #compute the model output
    output=model(input_images)
    loss=loss_function(output,target)
    model.zero_grad() # put all the previous gradients to 0 before backward pass

    # compute the gradient wrt the input_images
    loss.backward()
    input_images.requires_grad=False
    return input_images.grad.data

def pgd_attack(model, victim_image,loss_function,device, iters=40, eps=1e-2,
                    step_size=4/255, target=None, verbose=False):
    """
    Performing a PGD (projected gradient descent attack)
        args:
            model: the pytorch model on which to perform the attack
            victim_image: the image to attack
            loss_function: the loss function on which to compute the gradient
            device: cuda or cpu or mps
            iters: number of iterations of PGD
            eps: radius of the ball on which to project the gradient method
            step_size: step_size of one iteration of PGD
            target: the label (either towards which to go for targeted attacks
                or from which to maximise the step for untargeted attacks: target=None).
        returns:
            adv_images: attacked image
            pred_PGD: new prediction of the model
            confidence_attack_PGD: confidence in the attack
            loss_PGD: values of the loss during the PGD attack steps
    """
    adv_images = victim_image.clone().detach()
    if step_size*iters<eps:
        print("Warning, at best you will not be on the frontier of the unit eps ball")
    # first_pred
    loss_PGD=[]
    if target:
        # will be using gradient descent with the specified target
        if verbose:print('Running targeted FGSM')
        target_label=torch.tensor([target]).to(device)
    else :
        # will be using gradient ascent in order to maximise the loss
        # i.e. untergeted attack = gradient ascent
        if verbose:print('Running untargeted FGSM')
        with torch.no_grad():
            target_label=torch.tensor(model(victim_image).argmax()).reshape(1).to(device)
            # model.zero_grad()
    # computing the initial loss for the sake of comparison
    with torch.no_grad():
        adv_images=adv_images.detach()
        outputs = model(adv_images)
        initial_loss=loss_function(outputs,target_label)
    for i in range(iters) : 
        if verbose :print(f"Iteration {i+1}/{iters}", end="\r")
        adv_images=adv_images.detach()
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss_PGD.append(loss_function(outputs,target_label).item())
        model.zero_grad()
        gradient=compute_gradient(model=model,loss_function=loss_function,
                                    input_images=adv_images,target=target_label)
        if target:
            adv_images = adv_images - step_size*gradient.sign() 
        else:
            adv_images = adv_images + step_size*gradient.sign() 
        
        delta=torch.clamp(adv_images-victim_image,min=-eps,max=eps)
        adv_images=torch.clamp(victim_image+delta,min=0,max=1)
        # clamp = projection on the hypersphere
        # eta is the difference between new im and befor
        # eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    # evaluate how the attack performed
    with torch.no_grad():
        model.eval()
        output_adv_PGD=model(adv_images)
        soft_out=nn.Softmax(1)(output_adv_PGD)
        pred_PGD=soft_out.argmax()
        confidence_attack_PGD=soft_out.max().item()
        if verbose:
            print(f"Prediction class: {pred_PGD.item()}, with confidence {confidence_attack_PGD:.0%}")
            print(f"Initial loss was: {initial_loss}. New loss is {loss_function(output_adv_PGD,target_label)}")

    return adv_images,pred_PGD,confidence_attack_PGD,loss_PGD

def batch_pgd_attack(model,victim_images_labels,device,iters=PGD_NB_ITERS,eps=PGD_EPS, 
                    step_size=PGD_STEPSIZE,targets=None,verbose=False):
    """
    PGD attack on a batch of images, can be untergeted or targeted
    Warning : the loss function is a single scalar but is computed wrt to a whole batch
    This means that the variation in the loss for a single image is a minimal when using
    large batches
    params:
        victim_images: (images,labels) that should be attacked
        iters...
        targets: if None then running untargetted 
    """
    loss=nn.NLLLoss()
    victim_images, original_labels = victim_images_labels
    victim_images = victim_images.to(device)
    adv_images = victim_images.clone().detach()
    if step_size*iters<eps:
        print("Warning, at best you will not be on the frontier of the unit eps ball")
    # first_pred
    loss_PGD=[]
    if targets:
        # will be using gradient descent with the specified target
        if verbose:print('Running targeted FGSM')
        target_labels=torch.tensor([targets]).to(device)
    else :
        # will be using gradient ascent in order to maximimse the loss
        if verbose:print('Running untargeted FGSM')
        with torch.no_grad():
            target_labels=model(victim_images).argmax(dim=1).to(device)
            # model.zero_grad()
    for i in range(iters) : 
        if verbose :print(f"Iteration {i+1}/{iters}", end="\r")
        adv_images=adv_images.detach()
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss_PGD.append(loss(outputs,target_labels).item())
        model.zero_grad()
        gradient=compute_gradient(model=model,loss_function=loss,
                                        input_images=adv_images,target=target_labels)
        if targets:
            adv_images = adv_images - step_size*gradient.sign() 
        else:
            adv_images = adv_images + step_size*gradient.sign() 
        
        delta=torch.clamp(adv_images-victim_images,min=-eps,max=eps)
        adv_images=torch.clamp(victim_images+delta,min=0,max=1)
        # clamp = projection on the hypersphere
        # eta is the difference between new im and befor
        # eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        # images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    with torch.no_grad():
        model.eval()
        output_adv_PGD=model(adv_images)
        soft_out=nn.Softmax(dim=1)(output_adv_PGD)
        pred_PGD=soft_out.argmax(dim=1)
        confidence_attack_PGD=soft_out.max(dim=1).values
        if verbose:
            for i in range(len(pred_PGD)):
                print(f"Prediction class: {pred_PGD[i].item()},"\
                 f"with confidence {confidence_attack_PGD[i]:.0%}")
                print(f"New loss is {loss(output_adv_PGD[i],target_labels[i])}")

    return adv_images,pred_PGD,confidence_attack_PGD,loss_PGD

def compute_grad_finite_diff(net,single_image,label,loss,device,finite_epsilon):
    """
    Computing the gradient using finite difference on a single image
    in order to after run a blackbox attack. Grad_f~(f(x+e_i*epsilon)-f(x))/epsilon
    Where e_i is a canonical vector of the image vectorial space
    args:
        net: the model which to attack
        single_image: the image on which to compute the gradient
        label: label attached to the image (may be initial prediction or the real label)
        loss: loss function on which the model is trained
        device: torch device
        finite_epsilon: the small epsilon on which to compute the value of the gradient
    returns:
        torch tensor of the estimated gradient
    """
    finite_grad = torch.zeros((3,32,32)).to(device)
    net.eval()
    output_model = net(single_image.view(1,3,32,32))
    loss_image_init = loss(output_model,label)
    for i1 in range(3):
        for i2 in range(32):
            for i3 in range(32):
                e_i = torch.zeros((3,32,32)).to(device)
                e_i[i1,i2,i3]=1
                new_image = single_image + finite_epsilon * e_i
                new_image=new_image.to(device).view(1,3,32,32)
                net.eval()
                output_model = net(new_image)
                delta = (loss(output_model,label)- loss_image_init)
                finite_grad[i1,i2,i3] = delta/finite_epsilon
    return finite_grad

def finite_diff_attack(net,input_image,label,loss, finite_epsilon=1e-5):
    """
    Implementation of the finite_diff_attack: we compute the gradient using the function above using
    The finite difference approximation. Then we take a step of epsilon in the step of ascent
    args:
        net: the model which to attack
        input_image: the image on which to compute the gradient
        label: label attached to the image (may be initial prediction or the real label)
        loss: loss function on which the model is trained
        device: torch device
        finite_epsilon: the small epsilon on which to compute the value of the gradient
    returns:
        attacked_image: the image with the perturbation
    """
    finite_grad = compute_grad_finite_diff(net=net,single_image=input_image,label=label,
                                            loss=loss,device=device,finite_epsilon=finite_epsilon)
    finite_grad = finite_grad.to(device)
    attacked_image = torch.clamp(input_image + finite_grad,min=0,max=1)
    return attacked_image

def random_attack(victim_images_labels, epsilon, device, random_state):
  """
  Implementation of the random attack: takes a batch as input and returns a randomly perturbed
  version of it: image + epsilon*pert where pert is random uniform [-1,1]
  """
  torch.random.manual_seed(random_state)
  images, _ = victim_images_labels
  perturbation = (2*torch.rand(images.shape)-1)*epsilon
  perturbed_images = images + perturbation
  perturbed_images = perturbed_images.to(device)
  return perturbed_images

def mean_diff_attack(victim_images_labels, epsilon, device, random_state):
  """
  Implementation of mean_diff_attack: perturbs a batch of images in the direction of the class
  with closest mean
  """

  images, labels = victim_images_labels
  class_means = [torch.zeros([3,32,32])]*10
  class_counts = [0]*10
 
  # Compute class means
  for i in range(images.shape[0]):
    label = labels[i].item()
    image = images[i]
    class_means[label] = (class_counts[label]*class_means[label] + image) / (class_counts[label] + 1)
    class_counts[label] = class_counts[label] + 1
 
  # Perturb data
  perturbed_images = torch.zeros(images.shape)
 
  for i in range(images.shape[0]):
    label = labels[i].item()
    image = images[i]
    mean = class_means[label]  
    distances = [torch.norm(class_means[k] - mean) if k != label else np.inf for k in range(10)]
    target_mean = class_means[np.argmin(distances).item()]
    perturbation = epsilon*torch.sign(target_mean - mean)
    perturbed_image = image + perturbation
    perturbed_images[i] = perturbed_image
 
  perturbed_images = perturbed_images.to(device)
  return perturbed_images

if __name__=="__main__":
    from model import Net
    from CIFARLoader import CIFAR_Loader
    from settings import device,FINITE_DIFF_EPS


    loss_function=nn.NLLLoss()
    loader=CIFAR_Loader()
    model = Net()
    model.load("models/default_model.pth")
    model=model.to(device)
    #selecting a random image for sanity check
    test_loader = loader.get_test_loader()
    im_label=next(iter(test_loader))
    image=im_label[0][0].view(1,3,32,32).to(device)
    label=im_label[1][0].view(1).to(device)
    print("Testing the single image PGD attack")
    pgd_attack(model=model,victim_image=image,loss_function=loss_function,
        device=device,verbose=True)
    print("Testing the batch PGD attack")
    victim_images_labels = next(iter(test_loader))
    batch_pgd_attack(model=model, victim_images_labels= victim_images_labels,device=device,
                    iters=40,eps=1e-2, step_size=1/255,targets=None,verbose=False)
    print("Finished testing the PGD attacks")
    print("Testing finite diff blackbox attack")
    # finite_grad = compute_grad_finite_diff(net=model,single_image=image,label=label,loss=loss_function,
    #                                                 device=device,finite_epsilon=FINITE_DIFF_EPS)
    attacked_im_finite_grad = finite_diff_attack(net=model,input_image=image,label=label, loss=loss_function
                                                ,finite_epsilon=FINITE_DIFF_EPS)
    print(attacked_im_finite_grad.min().item(),attacked_im_finite_grad.max().item())
    print("Finished testing the blackbox finite gradient method")