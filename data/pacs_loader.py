import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from .download_pacs import download_and_extract_pacs

class PACSDataset(Dataset):
    def __init__(self, root_dir, domain, transform=None, train=True):
        """
        Args:
            root_dir (string): Directory with the PACS data
            domain (string): One of 'photo', 'art_painting', 'cartoon', 'sketch'
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): Whether to use train or test split
        """
        self.root_dir = root_dir
        self.domain = domain
        self.transform = transform
        self.train = train
        
        # Check if dataset exists and download if needed
        if not self._check_dataset_exists():
            print(f"PACS dataset not found at {root_dir}. Downloading...")
            success = download_and_extract_pacs(root_dir)
            if not success:
                raise ValueError(f"Failed to download PACS dataset to {root_dir}")
        
        # Define the class names in PACS
        self.classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Get all images for this domain
        self.image_paths = []
        self.labels = []
        
        domain_path = os.path.join(self.root_dir, self.domain)
        if not os.path.exists(domain_path):
            raise ValueError(f"Domain path {domain_path} does not exist!")
            
        for class_name in self.classes:
            class_path = os.path.join(domain_path, class_name)
            if not os.path.exists(class_path):
                continue
                
            for img_name in os.listdir(class_path):
                if img_name.endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        # If no data is found, raise error
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found for domain {domain}")
            
        # Split into train/test (80/20 split)
        indices = list(range(len(self.image_paths)))
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(0.8 * len(indices))
        
        if train:
            indices = indices[:split]
        else:
            indices = indices[split:]
            
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    def _check_dataset_exists(self):
        """Check if the PACS dataset exists at the specified location"""
        domains = ["photo", "art_painting", "cartoon", "sketch"]
        for domain in domains:
            domain_path = os.path.join(self.root_dir, domain)
            if not os.path.exists(domain_path):
                return False
        return True

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms(img_size=64):
    """
    Define transformations for the dataset
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform

def get_dataloader(config, domain, is_train=True):
    """
    Get dataloader for a specific domain
    
    Args:
        config: Configuration dict from config.yaml
        domain: The domain to load ('photo', 'art_painting', 'cartoon', 'sketch')
        is_train: Whether to use train or test data
    
    Returns:
        DataLoader for the specified domain
    """
    train_transform, test_transform = get_data_transforms(config['data']['img_size'])
    transform = train_transform if is_train else test_transform
    
    dataset = PACSDataset(
        root_dir=config['data']['data_path'],
        domain=domain,
        transform=transform,
        train=is_train
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['federated_learning']['batch_size'],
        shuffle=is_train,
        num_workers=2,
        pin_memory=True if config['federated_learning']['device'] == 'cuda' else False
    )
    
    return dataloader

def load_config(config_path='config/config.yaml'):
    """Load YAML config file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config 