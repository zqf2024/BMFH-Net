import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class dehaze_test_dataset(Dataset):
    def __init__(self, test_dir, test_name, tag=False):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = []
        for line in open(os.path.join(test_dir, 'test.txt')):
            line = line.strip('\n')
            if line != '':
                self.list_test.append(line)
        hazy, clean = test_name.split(',')
        self.tag = tag
        self.root_hazy = os.path.join(test_dir, '{}/'.format(hazy))
        self.root_clean = os.path.join(test_dir, '{}/'.format(clean))
        self.file_len = len(self.list_test)

    def __getitem__(self, index):
        if self.tag in ['thin', 'thick', 'moderation', 'RICE1', 'RICE2', 'RSID','NID','moderation_pt','RICE1_ALL','RICE2_ALL']:
            #test_name = self.list_test[index]  # .split('_')[0] + '.png'
             test_name = self.list_test[index].split('-')[0] + '-targets.png'
        else:
            test_name = self.list_test[index].split('-')[0] + '-targets.png'
        hazy = Image.open(os.path.join(self.root_hazy, self.list_test[index]))
        clean = Image.open(os.path.join(self.root_clean, test_name))

        hazy = self.transform(hazy)
        clean = self.transform(clean)

        return hazy, clean,test_name

    def __len__(self):
        return self.file_len
