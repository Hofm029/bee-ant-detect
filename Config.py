import torch
import numpy as np
import random

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

batch_size = 4
epochs = 5

# Paths
data_root = './hymenoptera_data/'
save_path = './weights_fine_tuning.pth'