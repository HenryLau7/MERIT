# -*- coding:utf-8 -*-

import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from augmentations import RandomCutout
from image_util import find_central
from copy import deepcopy

def train_val_split(dataroot, k, num_folds, rand_seed, cross_val):

    imgs = np.load(dataroot + "/imgs.npy", allow_pickle=True)
    labs = np.load(dataroot + "/labs.npy", allow_pickle=True)
    info = np.load(dataroot + "/info.npy", allow_pickle=True)

    random.seed(rand_seed)
    random.shuffle(imgs)
    random.seed(rand_seed)
    random.shuffle(labs)
    random.seed(rand_seed)
    random.shuffle(info)

    num_t = len(imgs) // num_folds
    
    if cross_val:
        if k == 0:
            train = {"imgs": imgs[num_t:], "labs": labs[num_t:]}
        elif k == num_folds - 1:
            train = {"imgs": imgs[:-num_t], "labs": labs[:-num_t]}
        else:
            train = {
                "imgs": np.concatenate((imgs[: num_t * k], imgs[num_t * (k + 1) :])),
                "labs": np.concatenate((labs[: num_t * k], labs[num_t * (k + 1) :])),
            }

        test = {
            "imgs": imgs[num_t * k : num_t * (k + 1)],
            "labs": labs[num_t * k : num_t * (k + 1)],
            "info": info[num_t * k : num_t * (k + 1)],
        }

        return (train, None, test)
    
    else:
        num_train = int(len(imgs) * 0.6)
        num_val = int(len(imgs) * 0.2)

        train_imgs = imgs[:num_train]
        train_labs = labs[:num_train]

        val_imgs = imgs[num_train:num_train + num_val]
        val_labs = labs[num_train:num_train + num_val]
        val_info = info[num_train:num_train + num_val]
        
        test_imgs = imgs[num_train + num_val:]
        test_labs = labs[num_train + num_val:]
        test_info = info[num_train + num_val:] 

        train = {"imgs": train_imgs, "labs": train_labs}
        val = {"imgs": val_imgs, "labs": val_labs, "info": val_info}
        test = {"imgs": test_imgs, "labs": test_labs, "info": test_info}

        return (train, val, test)


def test_cut(dataset, prior, rand_seed):
    if not isinstance(dataset, LiverDataset):
        raise ValueError("Dataset must be an instance of LiverDataset")
    if sum(prior) != 1:
        raise ValueError("Prior must sum to 1")

    labels = np.array([label for _, _, label, _ in dataset])
    total_samples = len(dataset)
    
    current_ratio, _ = get_dataset_stats(dataset)

    if current_ratio[0] == prior[0]:
        return dataset

    if current_ratio[0] > prior[0]:
        desired_count_1 = int(total_samples * current_ratio[1])
        desired_count_0 = max(1, int(desired_count_1 * prior[0] / prior[1]))
    else:
        desired_count_0 = int(total_samples * current_ratio[0])
        desired_count_1 = max(1, int(desired_count_0 * prior[1] / prior[0]))

    indices_0 = np.where(labels == 0)[0]
    indices_1 = np.where(labels == 1)[0]

    random.seed(rand_seed)
    random.shuffle(indices_0)
    random.seed(rand_seed)
    random.shuffle(indices_1)

    keep_indices_0 = indices_0[:desired_count_0]
    keep_indices_1 = indices_1[:desired_count_1]

    keep_indices = np.concatenate((keep_indices_0, keep_indices_1))
    keep_indices.sort()

    new_dataset = Subset(dataset, keep_indices)

    return new_dataset

class LiverDataset(Dataset):

    TASK_CONFIG = {
        1: lambda x: x > 2,   # S1-3 vs S4
        2: lambda x: x < 1,   # S1 vs S2-4
        3: lambda x: x > 1    # S1-2 vs S3-4
    }

    def __init__(self, data, transform, args, glb=True, train=True):
        
        self.imgs = self._process_images(data["imgs"])
        self.labs = self._process_labels(data["labs"], args.task)
        self.info = data.get("info", None)
        self.transform = transform
        self.glb = glb
        self.train = train

        # slide window args
        self.window_size = args.window_size
        self.patch_size = args.patch_size
        self.step_size = args.step_size

    def _process_images(self, images: np.ndarray) -> torch.Tensor:
        return np.concatenate([np.expand_dims(img, 0) for img in images], axis=0).astype(np.float32)
    
    def _process_labels(self, labels: np.ndarray, task: int) -> torch.Tensor:
        if task not in self.TASK_CONFIG:
            raise ValueError(f"Invalid task ID: {task}")
        return self.TASK_CONFIG[task](labels.astype(int))
    
    def __len__(self):
        return self.imgs.shape[0]
       
    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labs[idx]

        local_image, global_image = slide_window(image, window_size=self.window_size, k=self.patch_size, s=self.step_size)

        if self.train and self.transform:
            for i in range(local_image.shape[0]):
                local_image[i:i+1] = self.transform(local_image[i:i+1])

            if self.glb:
                global_image = self.transform(global_image)

        if self.train:
            return local_image, global_image, label
        else:
            info = self.info[idx]
            return local_image, global_image, label, info

def get_dataset(data, args, train=True):
    if train:
        transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),  
                    transforms.RandomVerticalFlip(),  
                    transforms.RandomRotation(30),  
                    transforms.ToTensor(), 
                    RandomCutout(min_size_ratio=0.02, max_size_ratio=0.15, max_crop=5, replacement=0),  
                ]
            )
    else:
        transform = None
        
    dataset = LiverDataset(data, transform, args, glb=True, train=train)
    return dataset

def get_dataloader(dataset, arg, train):
    dataloader = DataLoader(
        dataset,
        batch_size=arg.batch_size,
        shuffle=train,
        num_workers=arg.num_workers,
        pin_memory=True,
        prefetch_factor=4,
    )
    return dataloader

def get_dataset_stats(dataset):
    labels = [dataset[i][2] for i in range(len(dataset))]
    total_samples = len(labels)
    count_class_0 = labels.count(0)
    count_class_1 = labels.count(1)
    ratio = [count_class_0 / total_samples, count_class_1 / total_samples]
    return ratio, [count_class_0, count_class_1]

def slide_window(img, window_size, k, s):
    cent_x, cent_y = find_central(img)

    cent_x = min(max(cent_x, window_size // 2), img.shape[1] - window_size // 2)
    cent_y = min(max(cent_y, window_size // 2), img.shape[0] - window_size // 2)

    window = img[cent_y - window_size // 2:cent_y + window_size // 2, cent_x - window_size // 2:cent_x + window_size // 2]
    ims = torch.from_numpy(window).unsqueeze(0).unsqueeze(0)
    pas = ims.unfold(2, k, s).unfold(3, k, s)  # (1, 1, num_patches_y, num_patches_x, k, k)
    pas = pas.contiguous().view(1, -1, k, k)
    return pas[0], ims[0] # local views, global view

def load_datasets(args, fold: int):
    """Load and split datasets for current fold."""
    return train_val_split(args.dataroot, fold, args.rand_seed, args.cross_val)

def prepare_dataloaders(args, train_data, valid_data, test_data):
    train_set = get_dataset(train_data, args, train=True)
    test_set = test_cut(get_dataset(test_data, args, train=False), 
                       prior=args.test_cut, rand_seed=args.rand_seed) if args.test_cut else get_dataset(test_data, args, train=False)
    valid_set = get_dataset(valid_data, args, train=False) if not args.cross_val else deepcopy(test_set)

    train_prior = torch.tensor(get_dataset_stats(train_set)[0]) if args.train_prior else torch.tensor([0.5, 0.5])
    test_prior = torch.tensor(get_dataset_stats(test_set)[0]) if args.test_prior else torch.tensor([0.5, 0.5])

    return {
        "train": get_dataloader(train_set, args, train=True),
        "valid": get_dataloader(valid_set, args, train=False),
        "test": get_dataloader(test_set, args, train=False),
        "train_prior": train_prior,
        "test_prior": test_prior
    }