""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
import os
import pdb
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True ### 这是为了保证可以读入某些最后几个bytes被损坏了的图片 ###
from torch.utils.data import Dataset
import config
from utils import get_transform


image_path = {}
image_label = {}


class SDDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, phase='train', resize=(300,300)):
        assert phase in ['train', 'val', 'test']
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 150

        # get image path from images.txt
        with open(os.path.join(config.DATAPATH, 'images.txt')) as f:
            for line in f.readlines():
                line = line.strip()
                i_space = line.index(' ')
                id = line[:i_space]
                path = line[i_space+1:]
                #id, path = line.strip().split(' ')
                image_path[id] = path

        # get image label from image_class_labels.txt
        with open(os.path.join(config.DATAPATH, 'image_class_labels.txt')) as f:
            for line in f.readlines():
                id, label = line.strip().split(' ')
                image_label[id] = int(label)

        # get train/test image id from train_test_split.txt
        with open(os.path.join(config.DATAPATH, 'train_test_split.txt')) as f:
            for line in f.readlines():
                image_id, is_training_image = line.strip().split(' ')
                is_training_image = int(is_training_image)

                if self.phase == 'train' and is_training_image:
                    self.image_id.append(image_id)
                if self.phase in ('val', 'test') and not is_training_image:
                    self.image_id.append(image_id)

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]

        # image
        image = Image.open(os.path.join(config.Image_Dir, 'SD-merged', image_path[image_id])).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, image_label[image_id] # count begin from zero

    def __len__(self):
        return len(self.image_id)


if __name__ == '__main__':
    ds = SDDataset('train')
    print(len(ds))
    for i in range(0, 10):
        image, label = ds[i]
        print(image.shape, label)
