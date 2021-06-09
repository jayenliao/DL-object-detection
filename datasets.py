import torch, json, os
import torchvision.transforms.functional as FT
from torch.utils.data import Dataset
from PIL import Image
from utils import transform

class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, subset, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = subset.lower()
        assert self.split in ('tr', 'va', 'te', 'test')

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        if self.split != 'test':
            # Read data files
            with open(os.path.join(data_folder, 'images_'+ self.split + '.json'), 'r') as j:
                self.images = json.load(j)
            with open(os.path.join(data_folder, 'objects_'+ self.split + '.json'), 'r') as j:
                self.objects = json.load(j)
            assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        if self.split != 'test':
            # Read objects in this image (bounding boxes, labels, difficulties)
            objects = self.objects[i]
            boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
            labels = torch.LongTensor(objects['labels'])  # (n_objects)
            difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

            # Discard difficult objects, if desired
            if not self.keep_difficult:
                boxes = boxes[1 - difficulties]
                labels = labels[1 - difficulties]
                difficulties = difficulties[1 - difficulties]

            # Apply transformations
            image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)
            return image, boxes, labels, difficulties

        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
            new_image = FT.resize(image, dims=(300, 300))
            # Convert PIL image to Torch tensor
            new_image = FT.to_tensor(new_image)
            # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
            new_image = FT.normalize(new_image, mean=mean, std=std)
            return new_image


    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """
        images = []
        if self.split != 'test':
            boxes, labels, difficulties = [], [], []
            for b in batch:
                images.append(b[0])
                boxes.append(b[1])
                labels.append(b[2])
                difficulties.append(b[3])
            images = torch.stack(images, dim=0)
            return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each

        else:
            for b in batch:
                images.append(b[0])
            images = torch.stack(images, dim=0)
            return images
