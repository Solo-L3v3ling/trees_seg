from cProfile import label
import os
import shutil
from typing import Any, Dict, Optional, Tuple
from matplotlib import image
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torchvision.transforms.v2 as transforms
from torchvision.datasets import CocoDetection
from src.data.components.trees_dataset import TreesDataset
from src.utils.helpers import download_dataset_from_kaggle

class TreesDataModule(LightningDataModule):
    """
    LightningDataModule` for the Trees dataset.

    """

    def __init__(
            self,
            data_dir: str = "data/",
            sub_dataset: str = "trees",
            kaggle_datase_url: Optional[str] = None,
            train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `MNISTDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.Resize((256, 256)),

                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 256)),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dataset_path = f"{data_dir}/{sub_dataset}"
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.id2label = None
        self.label2id = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return len(self.hparams.id2label.keys())

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        print("Preparing data...")
    
        if self.hparams.kaggle_datase_url is not None and not os.path.exists(self.hparams.data_dir):
            path = download_dataset_from_kaggle(self.hparams.kaggle_datase_url)

            # Move the downloaded dataset to the data directory.
            shutil.move(path, self.dataset_path)


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:

            TREES_DATASET_DIR = self.dataset_path

            train_data = CocoDetection(root = TREES_DATASET_DIR + 'train/', annFile = TREES_DATASET_DIR + 'train/sem_annotations.coco.json')#, transform = self.train_transforms)
            test_data = CocoDetection(root = TREES_DATASET_DIR + 'test/', annFile = TREES_DATASET_DIR + 'test/sem_annotations.coco.json')#, transform = self.val_transforms)
            valid_data = CocoDetection(root = TREES_DATASET_DIR + 'valid/', annFile = TREES_DATASET_DIR + 'valid/sem_annotations.coco.json')#, transform = self.val_transforms)

            trainset = TreesDataset(dataset=train_data, transform=self.train_transforms)
            testset = TreesDataset(dataset=test_data, transform=self.val_transforms)
            validset = TreesDataset(dataset=valid_data, transform=self.val_transforms)
            # print(f"Trainset size: {len(trainset)}")
            # print(f"Testset size: {len(testset)}")
            # print(f"Validset size: {len(validset)}")
            # print(self.hparams.train_val_test_split)
            # Check for segmentation annotations

            dataset = ConcatDataset(datasets=[trainset, testset, validset])
            print(f"Dataset size: {len(dataset)}")
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

            self.label2id, self.id2label = trainset.label_id_map()



    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            collate_fn=collate_fn,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


def  collate_fn(batch):
    """Custom collate function to handle variable length sequences.

    :param batch: A batch of data.
    :return: A batch of data with variable length sequences handled.
    """
    print("Collate function called")
    images = [torch.tensor(item[0]) for item in batch]
    labels = [torch.tensor(item[1]) for item in batch]
    print(images[0].shape)
    print(labels[0].shape)
    print(len(images))
    print(len(labels))
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    print(images.shape)
    print(labels.shape)
    return images, labels
if __name__ == "__main__":
    _ = TreesDataModule()
