import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def eval(model_net, model_path, image_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model_net
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)
  model.eval()
  test_img = Image.open(image_path)
  test_img = np.array(test_img)
  test_img_t = np.rollaxis(test_img , 2)
  test_img_t = torch.tensor(test_img_t).unsqueeze(0).to(device).float()/255
  res = model(test_img_t)
  return res, test_img
