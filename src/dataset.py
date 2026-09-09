import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MaizeDataset(Dataset):
    """
    Maize Dataset class for loading both Dataset A (Variety) and Dataset B (Quality).
    Supports stratified splits and cross-validation folding.
    """
    def __init__(self, root_dir, label_mode="variety", split="train", seed=42, 
                 fold=None, num_folds=5, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            label_mode (str): 'variety' for Dataset A, 'quality' for Dataset B.
            split (str): 'train', 'val', or 'test'.
            seed (int): Random seed for reproducibility.
            fold (int): Specific fold index (0 to num_folds-1) for cross-validation on Dataset B.
            num_folds (int): Total number of folds for cross-validation.
            transform (callable, optional): Albumentations transform pipeline.
        """
        self.root_dir = root_dir
        self.label_mode = label_mode.lower()
        self.split = split
        self.seed = seed
        self.fold = fold
        self.num_folds = num_folds
        self.transform = transform

        if self.label_mode == "variety":
            self.classes = ["Bhihilifa", "SanzalSima", "WangDataa"]
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.samples = self._load_variety_samples()
        elif self.label_mode == "quality":
            self.classes = ["Bad", "Good"] # Bad = 0, Good = 1
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            self.samples = self._load_quality_samples()
        else:
            raise ValueError(f"Invalid label_mode: {label_mode}. Must be 'variety' or 'quality'.")

    def _load_variety_samples(self):
        """
        Loads and splits samples for Dataset A (Variety) using stratified split.
        """
        all_samples_by_class = {cls: [] for cls in self.classes}
        
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_path):
                raise FileNotFoundError(f"Class folder not found: {cls_path}")
            
            for f in sorted(os.listdir(cls_path)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(cls_path, f)
                    all_samples_by_class[cls].append((file_path, self.class_to_idx[cls]))
        
        # Perform deterministic stratified split (70/15/15)
        train_samples = []
        val_samples = []
        test_samples = []
        
        random.seed(self.seed)
        for cls in self.classes:
            samples = all_samples_by_class[cls]
            random.shuffle(samples)
            
            n = len(samples)
            idx_train = int(0.7 * n)
            idx_val = int(0.85 * n)
            
            train_samples.extend(samples[:idx_train])
            val_samples.extend(samples[idx_train:idx_val])
            test_samples.extend(samples[idx_val:])
            
        if self.split == "train":
            split_samples = train_samples
        elif self.split == "val":
            split_samples = val_samples
        elif self.split == "test":
            split_samples = test_samples
        else:
            raise ValueError(f"Invalid split name: {self.split}")
            
        # Shuffle the final subset
        random.shuffle(split_samples)
        return split_samples

    def _load_quality_samples(self):
        """
        Loads and splits samples for Dataset B (Quality) using stratified split or cross-validation.
        """
        all_samples_by_class = {cls: [] for cls in self.classes}
        
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            if not os.path.exists(cls_path):
                raise FileNotFoundError(f"Class folder not found: {cls_path}")
            
            for f in sorted(os.listdir(cls_path)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(cls_path, f)
                    all_samples_by_class[cls].append((file_path, self.class_to_idx[cls]))

        random.seed(self.seed)
        for cls in self.classes:
            random.shuffle(all_samples_by_class[cls])

        # If CV fold is specified, partition the dataset into folds
        if self.fold is not None:
            # Generate k-fold partitions stratified by class
            folds_by_class = {cls: [] for cls in self.classes}
            for cls in self.classes:
                samples = all_samples_by_class[cls]
                n = len(samples)
                fold_size = n // self.num_folds
                for i in range(self.num_folds):
                    start = i * fold_size
                    end = (i + 1) * fold_size if i < self.num_folds - 1 else n
                    folds_by_class[cls].append(samples[start:end])
            
            # Combine folds for train/val
            train_samples = []
            val_samples = []
            
            for i in range(self.num_folds):
                for cls in self.classes:
                    if i == self.fold:
                        val_samples.extend(folds_by_class[cls][i])
                    else:
                        train_samples.extend(folds_by_class[cls][i])
            
            # For cross validation, we don't have a separate test split in each fold (val is used for testing)
            if self.split == "train":
                split_samples = train_samples
            elif self.split in ["val", "test"]:
                split_samples = val_samples
            else:
                raise ValueError(f"Invalid split name: {self.split}")
        else:
            # Standard 70/15/15 stratified split
            train_samples = []
            val_samples = []
            test_samples = []
            
            for cls in self.classes:
                samples = all_samples_by_class[cls]
                n = len(samples)
                idx_train = int(0.7 * n)
                idx_val = int(0.85 * n)
                
                train_samples.extend(samples[:idx_train])
                val_samples.extend(samples[idx_train:idx_val])
                test_samples.extend(samples[idx_val:])
                
            if self.split == "train":
                split_samples = train_samples
            elif self.split == "val":
                split_samples = val_samples
            elif self.split == "test":
                split_samples = test_samples
            else:
                raise ValueError(f"Invalid split name: {self.split}")

        # Shuffle the final subset
        random.shuffle(split_samples)
        return split_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image with PIL
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Error loading image: {img_path}. Error: {e}")
            
        image_np = np.array(image)
        
        # Apply albumentations transforms
        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented['image']
        else:
            # Fallback tensor conversion
            image_np = image_np.transpose((2, 0, 1)) / 255.0  # HWC -> CHW
            image_tensor = torch.tensor(image_np, dtype=torch.float32)
            
        return image_tensor, label

def get_transforms(image_size=224):
    """
    Returns standard train and validation transforms.
    """
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform
