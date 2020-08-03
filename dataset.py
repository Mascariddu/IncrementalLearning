from torchvision.datasets import VisionDataset, CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np

class Cifar100(VisionDataset):
    def __init__(self, root, split, train, transform=None, target_transform=None):
        super(Cifar100, self).__init__(root, transform=transform, target_transform=target_transform)
        self.dataset = CIFAR100(root=root, train=train, download=True, transform=None)
        self.transform = transform
        self.pixel_mean = self.__getPixelMean__(root)
        
        '''shuffled_classes = [33, 29, 7, 71, 48, 53, 58, 80, 11, 91, 18, 84, 78, 36, 60,
                            1, 96, 90, 57, 54, 85, 17, 4, 92, 51, 99, 24, 95, 88, 89, 47,
                            22, 46, 12, 59, 19, 72, 82, 10, 26, 87, 68, 34, 39, 8, 16, 77,
                            21, 41, 97, 73, 38, 43, 63, 94, 9, 6, 2, 31, 14, 64, 15, 27, 23,
                            37, 45, 49, 74, 65, 83, 40, 75, 62, 50, 61, 79, 69, 81, 25, 66,
                            76, 3, 98, 30, 35, 5, 32, 52, 67, 20, 28, 0, 55, 13, 56, 42, 86,
                            44, 93, 70]'''

        if split == 9:
          shuffled_classes = [33, 29, 7, 71, 48, 53, 58, 80, 11, 91, 18, 84, 78, 36, 60, 1, 96, 90, 57, 54, 85, 17, 4, 92, 51, 99, 24, 95, 88, 89, 47, 22, 46, 12, 59, 19, 72, 82, 10, 26, 87, 68, 34, 39, 8, 16, 77, 21, 41, 97, 73, 38, 43, 63, 94, 9, 6, 2, 31, 14, 64, 15, 27, 23, 37, 45, 49, 74, 65, 83, 40, 75, 62, 50, 61, 79, 69, 81, 25, 66, 76, 3, 98, 30, 35, 5, 32, 52, 67, 20, 28, 0, 55, 13, 56, 42, 86, 44, 93, 70]
        elif split == 2:
          shuffled_classes = [14, 69, 22, 28, 55, 33, 85, 51, 8, 99, 94, 77, 90, 84, 35, 65, 60, 52, 0, 87, 59, 27, 92, 6, 70, 79, 21, 16, 47, 62, 3, 9, 49, 5, 44, 4, 57, 82, 74, 15, 89, 19, 48, 95, 50, 10, 7, 67, 46, 36, 34, 38, 53, 54, 23, 83, 37, 18, 39, 78, 73, 86, 96, 1, 58, 98, 20, 12, 97, 11, 29, 75, 81, 42, 56, 88, 24, 76, 66, 61, 31, 72, 13, 43, 2, 25, 40, 64, 71, 26, 93, 17, 63, 30, 45, 41, 80, 68, 91, 32]
        else:
          shuffled_classes = [2, 36, 38, 45, 23, 92, 40, 9, 48, 68, 42, 20, 31, 21, 62, 54, 93, 69, 14, 8, 7, 30, 49, 98, 26, 82, 77, 3, 46, 94, 43, 44, 75, 67, 66, 79, 17, 70, 15, 74, 6, 53, 76, 13, 10, 65, 55, 0, 56, 85, 91, 22, 5, 16, 86, 90, 28, 81, 29, 78, 18, 83, 99, 33, 84, 95, 59, 71, 1, 72, 24, 57, 4, 35, 60, 34, 73, 88, 97, 47, 12, 37, 27, 58, 50, 80, 25, 96, 87, 19, 89, 63, 32, 64, 41, 51, 52, 39, 11, 61]                   
        
        # Define classes per batch
        self.class_batches = dict.fromkeys(np.arange(10))
        for i in range(10):
            self.class_batches[i] = shuffled_classes[i*10:(i*10+10)]
        
        # Dictionary key:batch, value:batch_indexes
        self.batch_indexes = self.__BatchIndexes__()
        
        # Map labels from 0 to 99
        self.label_map = {k: v for v, k in enumerate(shuffled_classes)}
      

    def __getitem__(self, index):
        image, label = self.dataset[index]
        
        if self.transform is not None:
            image = self.transform(image)
            image -= self.pixel_mean
            
        return image, self.label_map[label]

    def __len__(self):
        return len(self.dataset)

    def __BatchIndexes__(self):
        batches = dict.fromkeys(np.arange(10))
        for i in range(10):
            batches[i] = []
        for idx,item in enumerate(self.dataset):
            for i in range(10):
                if item[1] in self.class_batches[i]:
                    batches[i].append(idx)
        return batches

    def __getBatchIndexes__(self,batch_index):
        return self.batch_indexes[batch_index]
    
    def __getPixelMean__(self,root):
        dataset = CIFAR100(root=root,train=True,transform=transforms.ToTensor())
        mean = torch.zeros((3,32,32))
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4, drop_last=False)
        for images, _ in loader:
            mean += sum(images)
        return mean / 50000
