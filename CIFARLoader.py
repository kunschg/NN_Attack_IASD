import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Dataset

class CIFAR_Loader(object):
    """
    Defines a class that helps creating both de datasets and the corresponding 
    dataloaders.
    args:
        root: root folder of the dataset. If the data hast yet been downloaded,
            it will then be in this folder
        frac: fraction of train set
    """
    def __init__(self,batch_size=32,train_eval_frac=0.9,
                    root_folder='data/') -> None:
        self.batch_size = batch_size
        self.train_eval_frac = train_eval_frac
        self.CIFAR_dataset = torchvision.datasets.CIFAR10(root = root_folder, 
                download = True, train = True, transform = transforms.ToTensor())
        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(self.CIFAR_dataset, 
                [round(train_eval_frac*len(self.CIFAR_dataset)), 
                        len(self.CIFAR_dataset) - round(train_eval_frac*len(self.CIFAR_dataset))])
        self.test_dataset=torchvision.datasets.CIFAR10(root = root_folder, 
                download = True, train = False, transform = transforms.ToTensor())

        self.train_loader=DataLoader(dataset=self.train_dataset,batch_size=batch_size,
                                        shuffle=True)
        self.test_loader=DataLoader(dataset=self.test_dataset,batch_size=batch_size,
                                        shuffle=False)
        self.validation_loader=DataLoader(dataset=self.validation_dataset,batch_size=batch_size,
                                        shuffle=False)
    def get_train_eval_datasets(self):
        return self.train_dataset, self.validation_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def get_train_eval_loaders(self):
        return self.train_loader, self.validation_loader

    def get_test_loader(self):
        return self.test_loader



