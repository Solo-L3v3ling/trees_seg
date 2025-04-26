import trace
from torch.utils.data import Dataset
import torch
import numpy as np


class TreesDataset(Dataset):
    def __init__(self, dataset,transform=None):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anno = self.dataset.coco.anns[idx]
        img = self.dataset[anno['image_id']][0]
        binary_masks = self.dataset.coco.annToMask(anno)
        if self.transform:
            target={
                'masks': binary_masks,

            }
            # img,binary_masks = self.transform(img,target)
            img = self.transform(img)
            binary_masks = self.transform(binary_masks)
        return img, binary_masks    #['masks'] if self.transform else binary_masks

    def label_id_map(self):
        return {}, {}