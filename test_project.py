#!/usr/bin/env python3

import os, os.path, sys
import argparse
import importlib 
import importlib.abc
import torch, torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from settings import device, batch_size
from CIFARLoader import CIFAR_Loader
from attacks import batch_pgd_attack

torch.seed()


def load_project(project_dir):
    module_filename = os.path.join(project_dir, 'model.py')
    if os.path.exists(project_dir) and os.path.isdir(project_dir) and os.path.isfile(module_filename):
        print("Found valid project in '{}'.".format(project_dir))
    else:
        print("Fatal: '{}' is not a valid project directory.".format(project_dir))
        raise FileNotFoundError 

    sys.path = [project_dir] + sys.path
    spec = importlib.util.spec_from_file_location("model", module_filename)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)
    return project_module

def test_natural(net, test_loader, num_samples):
    """
    Testing net model with natural images
    args:
        net: the model on which to evaluate
        test_loader:
        num_samples: the number of times to test a specific batch to get the accuracy
    """
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader,desc = "Natural test"):
            images, labels = data[0].to(device), data[1].to(device)
            for _ in range(num_samples):
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    return 100 * correct / total

def test_PGD_attacked_accuracy(net, test_loader,device,iters=40,eps=1e-2, 
                    step_size=1/255,num_samples=1):
    """
    Testing the net model with images generated with a PGD attack with the specified parameters
    """
    correct = 0
    total = 0

    for data in tqdm(test_loader, desc="Testing PGD mode"):
        images, labels = data[0].to(device), data[1].to(device)
        victim_images_labels = (images, labels)
        attacked_images, *_ = batch_pgd_attack(model=net, victim_images_labels=victim_images_labels,
                                            device=device,iters=iters,eps=eps,step_size=step_size,targets=None,verbose=False)
        for _ in range(num_samples):
            outputs = net(attacked_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", metavar="project-dir", nargs="?", default=os.getcwd(),
                        help="Path to the project directory to test.")
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Set batch size.")
    parser.add_argument("-s", "--num-samples", type=int, default=1,
                        help="Num samples for testing (required to test randomized networks).")
    parser.add_argument('-PGD_mode', '--PGD-mode', action="store_true",
                        help="Evaluate de model on images attacked by PGD")
    parser.add_argument('-model_path', '--model-path', type=str, default=None,
                        help="Train with a specific given model")

    args = parser.parse_args()
    project_module = load_project(args.project_dir)
    net = project_module.Net()
    net.to(device)
    if args.model_path:
        print(f"Testing the model: {args.model_path}")
        net.load(args.model_path)
    else:
        net.load_for_testing(project_dir=args.project_dir)

    loader = CIFAR_Loader(batch_size=batch_size,train_eval_frac=0.9,root_folder="data/")
    train_loader, valid_loader = loader.get_train_eval_loaders()
    
    if args.PGD_mode:
        acc_pgd = test_PGD_attacked_accuracy(net, test_loader= valid_loader,device=device)
        print(f"Model accuracy with PGD Attack : {acc_pgd}")


    acc_nat = test_natural(net, valid_loader, num_samples = args.num_samples)
    print("Model nat accuracy (test): {}".format(acc_nat))

if __name__ == "__main__":
    main()
