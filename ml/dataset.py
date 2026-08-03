import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .model import CLASS_NAMES

class JetbotDataset(Dataset):
    """
    Custom PyTorch Dataset for Multi-Class Autonomous Parking & Navigation.
    Loads images from a structured directory dataset/<class_name>/*.jpg.
    """
    def __init__(self, data_dir="dataset", is_train=True, image_size=(224, 224)):
        self.data_dir = data_dir
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self._load_dataset()

    def _load_dataset(self):
        if not os.path.exists(self.data_dir):
            return

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir, exist_ok=True)
                continue

            idx = self.class_to_idx[class_name]
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(class_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
