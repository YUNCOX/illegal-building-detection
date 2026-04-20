import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ChangeDetectionDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.baseline_dir = os.path.join(self.root_dir, 'baseline')
        self.recent_dir = os.path.join(self.root_dir, 'recent')
        self.mask_dir = os.path.join(self.root_dir, 'mask')
        
        # All directories should have the same filenames
        self.image_files = os.listdir(self.baseline_dir)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        baseline_path = os.path.join(self.baseline_dir, img_name)
        recent_path = os.path.join(self.recent_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load images
        baseline = Image.open(baseline_path).convert('RGB')
        recent = Image.open(recent_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # Grayscale for mask
        
        # Apply transforms
        baseline = self.transform(baseline)
        recent = self.transform(recent)
        mask = self.mask_transform(mask)
        
        # Binarize mask just to be safe
        mask = (mask > 0.5).float()
        
        return baseline, recent, mask
