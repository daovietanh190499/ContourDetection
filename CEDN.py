import torch
from torch import nn
from torchvision import models

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.vgg_features = models.vgg16_bn(pretrained=True).features
    # self.max_pool_idx = [4, 9, 16, 23, 30]
    self.max_pool_idx = [6, 13, 23, 33, 43]
    for idx in self.max_pool_idx:
      self.vgg_features[idx] = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    for p in self.vgg_features.parameters():
      p.requires_grad = False
    self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size = 3, stride=1, padding = 1)
  def forward(self, x):
    max_pool_info = []
    for idx, layer in enumerate(self.vgg_features):
      if idx not in self.max_pool_idx:
        x = layer(x)
      else:
        shape = x.shape
        x, ind = layer(x)
        max_pool_info.append({"kernel_size" : 2, "stride": 2, "padding": 0 ,"output_size": shape,"indices":ind})
    x = self.conv6(x)
    x = nn.functional.relu(x)
    x = nn.Dropout(p=0.2, inplace=True)(x)
    return x, max_pool_info

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.dconv6 = nn.Conv2d(in_channels = 4096, out_channels = 512, kernel_size = 1, stride=1)
    self.deconv5 = nn.ConvTranspose2d(in_channels = 512, out_channels = 512, kernel_size = 5, padding =2)
    self.deconv4 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 5 , padding = 2)
    self.deconv3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 5 ,padding = 2)
    self.deconv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 5 , padding = 2)
    self.deconv1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 5 ,padding = 2)
    self.pred = nn.ConvTranspose2d(in_channels = 32, out_channels = 1, kernel_size = 5, padding = 2)
  
  def forward(self, x, max_pool_info):
    x = self.dconv6(x)
    x = nn.functional.relu(x)
    x = nn.Dropout(p=0.2, inplace=True)(x)
    x = nn.functional.max_unpool2d(x, **max_pool_info[4])

    x = self.deconv5(x)
    x = nn.functional.relu(x)
    x = nn.Dropout(p=0.2, inplace=True)(x)
    x = nn.functional.max_unpool2d(x, **max_pool_info[3])

    x = self.deconv4(x)
    x = nn.functional.relu(x)
    x = nn.Dropout(p=0.2, inplace=True)(x)
    x = nn.functional.max_unpool2d(x, **max_pool_info[2])

    x = self.deconv3(x)
    x = nn.functional.relu(x)
    x = nn.Dropout(p=0.2, inplace=True)(x)
    x = nn.functional.max_unpool2d(x, **max_pool_info[1])

    x = self.deconv2(x)
    x = nn.functional.relu(x)
    x = nn.Dropout(p=0.2, inplace=True)(x)
    x = nn.functional.max_unpool2d(x, **max_pool_info[0])

    x = self.deconv1(x)
    x = nn.functional.relu(x)
    x = nn.Dropout(p=0.2, inplace=True)(x)

    x = self.pred(x)
    x = torch.sigmoid(x)
    return x

class CEDN(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self,x):
    x, max_pool_info = self.encoder(x)
    return self.decoder(x, max_pool_info)

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  x = torch.rand(8, 3, 500, 500)
  x = x.to(device)
  model = CEDN()
  model.to(device)
  model.eval()
  y = model(x)
  print(y.shape)
  print(model)
