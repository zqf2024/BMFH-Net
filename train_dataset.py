import os
import random

import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# data augmentation for image rotate
def augment(hazy, clean ):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        hazy = transforms.functional.rotate(hazy, rotate_degree)
        clean = transforms.functional.rotate(clean, rotate_degree)

        return hazy, clean
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        hazy = vertical_flip(hazy)
        clean = vertical_flip(clean)

        return hazy, clean
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        hazy = horizontal_flip(hazy)
        clean = horizontal_flip(clean)

        return hazy, clean
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return hazy, clean


class dehaze_train_dataset(Dataset):
    def __init__(self, train_dir, train_name, tag=False):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_train = []
        for line in open(os.path.join(train_dir, 'train.txt')):
            line = line.strip('\n')
            if line != '':
                self.list_train.append(line)
        hazy, clean = train_name.split(',')
        self.tag = tag
        self.root_hazy = os.path.join(train_dir, '{}/'.format(hazy))
        self.root_clean = os.path.join(train_dir, '{}/'.format(clean))
        self.file_len = len(self.list_train)

    def __getitem__(self, index):
        name = self.list_train[index]  # .split('_')[0] + '.jpg'
        hazy = Image.open(os.path.join(self.root_hazy, name))
        clean = Image.open(os.path.join(self.root_clean, name))

        # crop a patch
        i, j, h, w = transforms.RandomCrop.get_params(hazy, output_size=(256, 256))
        hazy_ = TF.crop(hazy, i, j, h, w)
        clean_ = TF.crop(clean, i, j, h, w)


        # data argumentation
        hazy_arg, clean_arg = augment(hazy_, clean_)
        hazy = self.transform(hazy_arg)
        clean = self.transform(clean_arg)

        return hazy, clean

    def __len__(self):
        return self.file_len
