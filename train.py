from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import models
from torchvision import transforms
from torchvision.transforms import functional as tvf
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import time
import os
import argparse
import tqdm
import sys
import random

class CustomDataset(Dataset):
  def __init__(self, image_dir, label_dir, imgs_ann, mode, aug_mode):
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.mode = mode
    self.aug_mode = aug_mode
    with open(imgs_ann, 'r') as f:
      self.imgs = f.read().splitlines()

  def transform(self, image_origin, mask_origin, mode, data_augmentation = "randomcrop"):
        image_res, mask_res = None, None
        totensor_op = transforms.ToTensor()
        color_op = transforms.ColorJitter(0.1, 0.1, 0.1)
        resize_op = transforms.Resize((224, 224))
        image_origin = color_op(image_origin)
      #  norm_op = transforms.Normalize(mean =[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        if data_augmentation == 'randomcrop':
            if image_origin.size[0] < 224 or image_origin.size[1] < 224:
                #padding-val:
                val = int(np.array(image_origin).sum() / image_origin.size[0] / image_origin.size[1])
                padding_width = 224-min(image_origin.size[0],image_origin.size[1])
                padding_op = transforms.Pad(padding_width,fill=val)
                image_origin = padding_op(image_origin)
                padding_op = transforms.Pad(padding_width, fill=0)
                mask_origin = padding_op(mask_origin)
            i, j, h, w = transforms.RandomCrop.get_params(
                image_origin, output_size=(224, 224)
            )
            image_res = totensor_op(tvf.crop(image_origin, i, j, h, w))
            mask_res = totensor_op(tvf.crop(mask_origin, i, j, h, w))

        elif data_augmentation == 'resize':
            image_res = totensor_op(resize_op(image_origin))
            mask_res = totensor_op(resize_op(mask_origin))
      #  image_res = norm_op(image_res)
        # mask_res = mask_res/255
     
        return image_res, mask_res

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    input_img = Image.open(self.image_dir + self.imgs[idx] + '.jpg').convert('RGB')
    label_img = Image.open(self.label_dir + self.imgs[idx] + '.jpg')
    input, label = self.transform(input_img, label_img, self.mode, self.aug_mode)
    return input, label

class Trainer:
  def __init__(self, model):
    self.images_path = ""
    self.ctns_path = ""
    self.train_path = ""
    self.val_path = ""
    self.model_save_path = ""
    self.model_save_name = ""
    self.train_losses = []
    self.test_losses = []
    self.train_accuracies = []
    self.test_accuracies = []
    self.start_epoch = 1
    self.max_epoch = 100
    self.save_epoch_freq = 1
    self.save_iter_freq = 50
    self.print_freq = 10
    self.batch_size = 64
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = model
    if self.model_save_path and self.model_save_name:
      self.model.load_state_dict(torch.load(self.model_save_path + self.model_save_name))
    self.model.to(self.device)
    print("device: ", self.device)
    self.model.train()
    self.lr = 1e-4
    self.optimizer = torch.optim.Adam([x for x in list(self.model.parameters()) if x.requires_grad == True], lr=self.lr)
    self.bce =  nn.BCELoss(reduction = "none")
    self.mse = nn.MSELoss(reduction = "none")
    self.criterion = F.binary_cross_entropy
    self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.98)
    
  def set_config(
    self,
    lr=1e-4, 
    batch_size=64,
    start_epoch=30,
    max_epoch = 100,
    images_path="",
    ctns_path="",
    train_path="",
    val_path="",
    model_save_path="",
    model_save_name=""
  ):
    self.lr=lr
    self.batch_size=batch_size 
    self.images_path = images_path
    self.ctns_path = ctns_path
    self.train_path = train_path
    self.val_path = val_path
    self.model_save_path = model_save_path
    self.model_save_name = model_save_name
    self.start_epoch = start_epoch
    self.max_epoch = max_epoch
    if images_path != "" or ctns_path != "" or train_path != "" or val_path != "":
      self.iniloader = DataLoader(CustomDataset(self.images_path, self.ctns_path, self.train_path, mode='train', aug_mode='resize'), batch_size=self.batch_size)
      self.trainloader = DataLoader(CustomDataset(self.images_path, self.ctns_path, self.train_path, mode='train', aug_mode='randomcrop'), batch_size=self.batch_size)
      self.testloader = DataLoader(CustomDataset(self.images_path, self.ctns_path, self.val_path, mode='val', aug_mode='randomcrop'), batch_size=self.batch_size)


  def loss(self,outputs, targets):
    weights = torch.empty_like(targets).to(self.device)
    weights[targets >= .97] = 10
    weights[targets < .97] = 1
    loss = self.criterion(outputs, targets, weights)
    return loss

  def show_dataloader(self):
    dataiter = iter(self.trainloader)
    images, labels = dataiter.next()

    img = torchvision.utils.make_grid(images)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    img = torchvision.utils.make_grid(labels)
    npimg = img.numpy()
    plt.imshow(npimg[0])
    # unique, counts = np.unique(npimg[0, :224, :224], return_counts=True)
    # print(dict(zip(unique, counts)))
    # plt.show()
    # plt.imshow(npimg[0, :224, :224])
  
  def loss_plot(self):
    # plt.plot(train_accuracies, label='Training accuracy')
    # plt.plot(test_accuracies, label='Validation accuracy')
    # plt.legend(frameon=False)
    # plt.show()
    plt.plot(self.train_losses, label='Training loss')
    plt.plot(self.test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()

  def train(self):
    epochs = (self.start_epoch, self.max_epoch)
    running_loss = 0
    train_loss = 0
    running_accuracy = 0
    train_accuracy = 0
    print_every = self.print_freq
    # print("Start load image! please wait until train start")
    #for steps, (inputs, labels) in enumerate(self.iniloader):
    #  if int((steps*100)/(len(self.iniloader) - 1))%10 == 0:
    #    print(str(steps), end="___")
    #  if steps == len(self.iniloader) - 1:
    #    print(' ')
    # print("End load image")
    for epoch in range(epochs[0], epochs[1]):
      for steps, (inputs, labels) in enumerate(self.trainloader):
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        logps = self.model.forward(inputs)
        loss = self.loss(logps, labels)
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
        train_loss += loss.item()

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        running_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        if steps % print_every == 0:
          print(f"Epoch [{epoch+1}|{epochs[1]}] "
              f"Iter [{steps}|{len(self.trainloader)}] "
              f"Train loss: {running_loss/print_every:.3f} ")
              # f"Train accuracy: {running_accuracy/print_every:.3f}")
          if steps % self.save_iter_freq == 0 and steps != 0:
            print("Saving state ...")
            torch.save(self.model, self.model_save_path + 'cedn_iter_' + str(steps) +'.pth')
          running_loss = 0
          running_accuracy = 0

      print("Calculate val loss ...")
      test_loss = 0
      test_accuracy = 0
      self.model.eval()
      with torch.no_grad():
        for inputs, labels in self.testloader:
          inputs, labels = inputs.to(self.device), labels.to(self.device)
          logps = self.model.forward(inputs)
          batch_loss = self.loss(logps, labels)
          test_loss += batch_loss.item()
          
          ps = torch.exp(logps)
          top_p, top_class = ps.topk(1, dim=1)
          equals = top_class == labels.view(*top_class.shape)
          test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        self.train_losses.append(train_loss/len(self.trainloader))
        self.test_losses.append(test_loss/len(self.testloader))
        self.train_accuracies.append(train_accuracy/len(self.trainloader))
        self.test_accuracies.append(test_accuracy/len(self.testloader))

        print(f"Epoch [{epoch+1}|{epochs[1]}] "
              f"Train loss: {train_loss/len(self.trainloader):.3f} "
              # f"Train accruracy: {train_accuracy/len(self.trainloader):.3f} "
              f"Test loss: {test_loss/len(self.testloader):.3f} ")
              # f"Test accuracy: {test_accuracy/len(self.testloader):.3f}")
        self.model.train()
        train_loss = 0
        train_accuracy = 0
      self.scheduler.step()

      if epoch % self.save_epoch_freq == 0:
        print("Saving state ...")
        torch.save(self.model.state_dict(), self.model_save_path + 'cedn_epoch_' + str(epoch + 1) +'.pth')
