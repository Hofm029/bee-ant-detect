import glob
import os.path as osp
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from PIL import Image

from Config import size, mean, std, save_path, data_root
from ImageTranform import ImageTransform

def make_datapath_list(phase='train'):
  target_path = osp.join(data_root+phase+'/**/*.jpg')

  path_list = []
  for path in glob.glob(target_path):
    path_list.append(path)

  if len(path_list) == 0:
    raise FileNotFoundError(f"No images found in {target_path}")

  return path_list

def train_model(net, dataloader_dict, criterion, optimizer, epochs):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("device: ", device)

  for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs))

    net.to(device)

    torch.backends.cudnn.benchmark = True

    for phase in ['train', 'val']:
      if phase == 'train':
        net.train()
      else:
        net.eval()

      epoch_loss = 0.0
      epoch_point = 0

      for inputs, labels in tqdm(dataloader_dict[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = net(inputs)
          loss = criterion(outputs, labels)
          _, preds = torch.max(outputs, 1)

          if phase == 'train':
            loss.backward()
            optimizer.step()
        
          epoch_loss += loss.item()*inputs.size(0) #lay batch size (input.size(0))
          epoch_point += torch.sum(preds==labels.data)

      epoch_loss = epoch_loss/len(dataloader_dict[phase].dataset)
      epoch_accuracy = epoch_point.double() / len(dataloader_dict[phase].dataset)
      print("-+-+-+-+-+-+-+-+-=========-+-+-+-+-+-+-+-+-+-+")
      print('{} Loss: {:4f} Acc: {:4f}'.format(phase, epoch_loss, epoch_accuracy))
  
  torch.save(net.state_dict(), save_path)

def params_to_update(net):
    params_to_update_1 = []
    params_to_update_2 = []
    params_to_update_3 = []

    update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
      # Features layer (all params starting with 'features.')
      if name.startswith("features"):
        param.requires_grad = True
        params_to_update_1.append(param)
      # Classifier intermediate layers
      elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
      # Classifier final layer
      elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
      else:
        param.requires_grad = False
    return params_to_update_1, params_to_update_2, params_to_update_3

def load_model(net, model_path):
  try:
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights)
  except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found at {model_path}")
  except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {e}")

  # load tu gpu sang cpu
  # load_weights = torch.load(model_path, map_location={'cuda:0': 'cpu'})
  # net.load_state_dict(load_weights)
  return net

class Predictor():
  """Predictor class that reuses model for multiple predictions"""
  def __init__(self, class_index, model_path=None):
    self.class_index = class_index
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    self.net = models.vgg16(pretrained=True)
    self.net.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
    self.net.eval()

    # Load trained weights if provided
    if model_path:
      load_model(self.net, model_path)

    self.net.to(self.device)

    # Initialize transforms
    self.transforms = ImageTransform(size, mean, std)

  def predict(self, img):
    """Predict class for a single image (PIL Image)"""
    # Transform image
    img_transformed = self.transforms(img, phase='test')
    img_transformed = img_transformed.unsqueeze(0).to(self.device)

    # Predict
    with torch.no_grad():
      output = self.net(img_transformed)
      probabilities = torch.softmax(output, dim=1)
      max_id = torch.argmax(probabilities, dim=1).item()
      confidence = probabilities[0][max_id].item()

    predicted_label = self.class_index[max_id]
    return predicted_label, confidence

def predict(img):
  """Legacy predict function for backward compatibility"""
  predictor = Predictor(['ants', 'bees'], model_path=save_path)
  label, confidence = predictor.predict(img)
  print(f"Predicted: {label} (confidence: {confidence:.2%})")
  return label
