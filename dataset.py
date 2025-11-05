import os
from PIL import Image
import torch.utils.data as data

class MyDataset(data.Dataset):
  def __init__(self, file_list, transforms=None, phase='train'):
    self.file_list = file_list
    self.transforms = transforms
    self.phase = phase

  def __len__(self):
    return len(self.file_list)

  def __getitem__(self, index):
    img_path = self.file_list[index]

    try:
      img = Image.open(img_path)
    except (IOError, OSError) as e:
      raise RuntimeError(f"Failed to load image at {img_path}: {e}")

    img_transforms = self.transforms(img, self.phase)

    # Extract label from directory name (e.g., 'ants' or 'bees')
    label_name = os.path.basename(os.path.dirname(img_path))

    if label_name == 'ants':
      label = 0
    elif label_name == 'bees':
      label = 1
    else:
      raise ValueError(f"Unknown label: {label_name}")

    return img_transforms, label