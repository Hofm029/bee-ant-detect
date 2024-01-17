from lib import *
from ImageTranform import ImageTransform
from Config import *
from utils import make_datapath_list, train_model, params_to_update, load_model, predict
from dataset import MyDataset

def __main__():
  train_list = make_datapath_list('train')
  val_list = make_datapath_list('val')
  #dataset
  train_dataset = MyDataset(train_list, transforms=ImageTransform(size, mean, std), phase='train')
  val_dataset = MyDataset(val_list, transforms=ImageTransform(size, mean, std), phase='val')

  #data loader
  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)

  dataloader_dict = {"train": train_dataloader, "vali": val_dataloader}   

  #network    
  use_pretrained = True
  net = models.vgg16(pretrained=use_pretrained)
  net.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

  #setting mode
  #net.train()

  #loss function
  criterior = nn.CrossEntropyLoss()

  #transfer learning
  params_1, params_2, params_3 = params_to_update(net)
  
  #optimizer  
  optimizer = optim.SGD([
    {'params': params_1, 'lr': 1e-4},
    {'params': params_2, 'lr': 5e-4},
    {'params': params_3, 'lr': 1e-3}
    ], momentum=0.9
    )

  #training
  train_model(net, dataloader_dict, criterior, optimizer, epochs)

if __name__ == "__main__":
  # __main__()

  # #network
  # use_pretrained = True
  # net = models.vgg16(pretrained=use_pretrained)
  # net.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)

  # load_model(net, save_path)
  img_show = cv2.imread('./hymenoptera_data/hymenoptera_data/train/bees/1097045929_1753d1c765.jpg')
  cv2.imshow('img', img_show)
  cv2.waitKey(0)
  img = Image.open('./hymenoptera_data/hymenoptera_data/train/bees/1097045929_1753d1c765.jpg')
  predict(img)