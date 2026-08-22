import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from .model import CLASS_NAMES

# Maps existing dataset folder names -> new class indices.
# parking_spot_occupied is merged into blocked (idx 3) because from a
# front-facing camera on a small robot an occupied spot and a blocked path
# are visually indistinguishable and the distinction adds training noise.
FOLDER_TO_CLASS = {
    "path_free":             0,  # -> lane_clear
    "parking_spot_left":     1,  # -> spot_left
    "parking_spot_right":    2,  # -> spot_right
    "obstacle_blocked":      3,  # -> blocked
    "parking_spot_occupied": 3,  # -> blocked (merged)
}

# Which class index is the left/right pair for label-swap on horizontal flip.
FLIP_SWAP = {1: 2, 2: 1}  # spot_left <-> spot_right


class JetbotDataset(Dataset):
    """
    Custom PyTorch Dataset for 4-class Autonomous Parking.

    Key improvements over v1:
    - Folder remapping: merges parking_spot_occupied into blocked
    - Horizontal flip with label swap: flipping a spot_left image gives a
      spot_right image for free. This doubles left/right data and forces the
      model to learn SPATIAL asymmetry (which side of the frame has the gap)
      rather than texture/colour shortcuts.
    - Random erasing: simulates partial occlusion
    - Stronger colour jitter to generalise across lighting conditions
    """

    def __init__(self, data_dir="dataset", is_train=True, image_size=(224, 224)):
        self.data_dir = data_dir
        self.is_train = is_train
        self.image_size = image_size
        self.samples = []  # list of (path, new_class_idx)

        self._load_dataset()

        print(f"[Dataset] Loaded {len(self.samples)} samples from '{data_dir}'")
        self._print_class_distribution()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.is_train:
            image, label = self._train_transform(image, label)
        else:
            image = self._val_transform(image)

        return image, label

    def _load_dataset(self):
        if not os.path.exists(self.data_dir):
            return

        for folder_name, class_idx in FOLDER_TO_CLASS.items():
            class_dir = os.path.join(self.data_dir, folder_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

    def _print_class_distribution(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        for idx, name in enumerate(CLASS_NAMES):
            print(f"  [{idx}] {name:<12}: {counts.get(idx, 0)} samples")

    def _train_transform(self, image, label):
        """
        Augmentation pipeline with label-aware horizontal flip.
        Horizontal flip swaps spot_left <-> spot_right labels so the model
        learns that left/right is purely about spatial position in the frame.
        """
        # 1. Resize
        image = TF.resize(image, list(self.image_size))

        # 2. Horizontal flip with label swap
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = FLIP_SWAP.get(label, label)  # swap left/right; others unchanged

        # 3. Random affine (slight rotation + translate to simulate robot wobble)
        if random.random() < 0.4:
            angle = random.uniform(-8, 8)
            translate = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
            image = TF.affine(image, angle=angle, translate=[int(t * self.image_size[0]) for t in translate],
                              scale=1.0, shear=0)

        # 4. Colour jitter (lighting variation)
        colour_jitter = transforms.ColorJitter(
            brightness=0.35, contrast=0.35, saturation=0.30, hue=0.12
        )
        image = colour_jitter(image)

        # 5. Random grayscale (5% chance — simulates low-light / IR camera)
        if random.random() < 0.05:
            image = TF.to_grayscale(image, num_output_channels=3)

        # 6. ToTensor + ImageNet normalise
        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # 7. Random erasing (simulates partial occlusion by wires/objects near camera)
        eraser = transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0)
        image = eraser(image)

        return image, label

    def _val_transform(self, image):
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(image)
