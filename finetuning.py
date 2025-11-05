import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models

from ImageTranform import ImageTransform
from Config import size, mean, std, batch_size, epochs, save_path
from utils import make_datapath_list, train_model, params_to_update, Predictor
from dataset import MyDataset

def train():
  """Train the model"""
  print("Starting training...")

  # Load data
  train_list = make_datapath_list('train')
  val_list = make_datapath_list('val')

  # Create datasets
  train_dataset = MyDataset(train_list, transforms=ImageTransform(size, mean, std), phase='train')
  val_dataset = MyDataset(val_list, transforms=ImageTransform(size, mean, std), phase='val')

  # Create data loaders
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

  dataloader_dict = {"train": train_dataloader, "val": val_dataloader}

  # Initialize network
  use_pretrained = True
  net = models.vgg16(pretrained=use_pretrained)
  net.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

  # Loss function
  criterion = nn.CrossEntropyLoss()

  # Transfer learning - configure layer-wise learning rates
  params_1, params_2, params_3 = params_to_update(net)

  # Optimizer with differential learning rates
  optimizer = optim.SGD([
    {'params': params_1, 'lr': 1e-4},
    {'params': params_2, 'lr': 5e-4},
    {'params': params_3, 'lr': 1e-3}
    ], momentum=0.9
    )

  # Train model
  train_model(net, dataloader_dict, criterion, optimizer, epochs)
  print(f"Training completed! Model saved to {save_path}")

def inference(image_path):
  """Run inference on a single image"""
  print(f"Running inference on {image_path}...")

  # Load image
  img = Image.open(image_path)

  # Create predictor and predict
  predictor = Predictor(['ants', 'bees'], model_path=save_path)
  label, confidence = predictor.predict(img)

  print(f"Prediction: {label}")
  print(f"Confidence: {confidence:.2%}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Bee-Ant Classification with VGG16')
  parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'],
                      help='Mode: train or inference')
  parser.add_argument('--image', type=str, default=None,
                      help='Path to image for inference mode')

  args = parser.parse_args()

  if args.mode == 'train':
    train()
  elif args.mode == 'inference':
    if args.image is None:
      print("Error: --image argument is required for inference mode")
      parser.print_help()
    else:
      inference(args.image)