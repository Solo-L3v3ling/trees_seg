from torch.utils.data import Dataset


class TreesDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anno = self.dataset.coco.anns[idx]
        img = self.dataset[anno['image_id']][0]
        binary_masks = self.dataset.coco.annToMask(anno)
        return img, binary_masks

    def label_id_map(self):
        return {}, {}