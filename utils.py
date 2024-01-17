from lib import * 
from Config import *
from ImageTranform import ImageTransform
def make_datapath_list(phase='train'):
  rootpath = './hymenoptera_data/'
  targer_path = osp.join(rootpath+phase+'/**/*.jpg')

  path_list = []
  for path in glob.glob(targer_path):
    path_list.append(path)

  return path_list

def train_model(net, dataloader_dict, criterior, optimizer, epochs):
  devide = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("devide: ", devide)

  for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch, epochs))

    net.to(devide)

    torch.backends.cudnn.benchmark = True

    for phase in ['train', 'vali']:
      if phase == 'train':
        net.train()
      else:
        net.eval()

      epoch_loss = 0.0
      epoch_point = 0
      if (epoch == 0) and (phase == 'train'):
        continue
      for inputs, labels in tqdm(dataloader_dict[phase]):
        inputs = inputs.to(devide)
        labels = labels.to(devide)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs = net(inputs)
          loss = criterior(outputs, labels)
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

    update_param_names_1 = ["features"]
    update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
    update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

    for name, param in net.named_parameters():
      if name in update_param_names_1:
        param.requires_grad = True
        params_to_update_1.append(param)
      elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
      elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
      else:
        param.requires_grad = False
    return params_to_update_1, params_to_update_2, params_to_update_3

def load_model(net, model_path):
  load_weights = torch.load(model_path)
  net.load_state_dict(load_weights)
  # print(net)
  # for name, param in net.named_parameters():
  #   print(name, param)
  # load tu gpu sang cpu
  # load_weights = torch.load(model_path, map_location=('cuda:0': 'cpu'))
  # net.load_state_dict(load_weights)
  return net

class Predictor():
  def __init__(self, class_index):
    self.class_index = class_index

  def predict_max(self, out):
    max_id = np.argmax(out.detach().numpy())
    predicted_label_name = self.class_index[max_id]
    return predicted_label_name
  
def predict(img):
  #network
  use_pretrained = True
  net = models.vgg16(pretrained=use_pretrained)
  net.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
  net.eval()

  #model
  load_model(net, save_path)

  #transform
  transforms = ImageTransform(size, mean, std)
  img_transformed = transforms(img, phase='test')
  img_transformed = img_transformed.unsqueeze_(0)

  #predict
  predictor = Predictor(['ants', 'bees'])
  output = net(img_transformed)
  response = predictor.predict_max(output)
  print(response)
