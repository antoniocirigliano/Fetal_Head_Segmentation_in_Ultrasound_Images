# 23 week, SP and RP plans

import os
import warnings
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision

import Plot2

# GDX

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.getcwd()

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Model

Encoder = 'timm-efficientnet-b0'
weights = 'imagenet'

from torch import nn
import segmentation_models_pytorch as sample_data
from segmentation_models_pytorch.losses import DiceLoss

class SegmentationModel(nn.Module):

  def __init__(self):
    super(SegmentationModel, self).__init__()

    self.arc=sample_data.Unet(
        encoder_name = Encoder,
        encoder_weights = weights,
        in_channels = 3, 
        classes = 1,
        activation = None
    )

  def forward(self, images, masks = None):

    logits = self.arc(images)

    if masks != None:

      loss1 = DiceLoss(mode='binary')(logits, masks)
      loss2 = nn.BCEWithLogitsLoss()(logits, masks)
      return logits, loss1 + loss2
    
    return logits

model = SegmentationModel()
model.to(DEVICE)


# Set the path to your folder of images
folder_path = '/home/antonio/workspace/ucl-thesis/data/SPs'

# Get the list of image files in the folder
image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]

# Create a Pandas DataFrame with the file paths
df = pd.DataFrame({'images': image_files})

# Save the DataFrame to a CSV file
df.to_csv('SPs.csv', index=False)

CSV = '/home/antonio/workspace/ucl-thesis/SPs.csv'

#df = pd.read_csv(CSV)
#df.head()

class SegmentationData(Dataset):

  def __init__(self, df):
    self.df = df
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    row = self.df.iloc[idx]
    image_path = row.images
    image = cv2.imread(image_path)

    # resize the image to 320x320
    image = cv2.resize(image, (320, 320))

    #(h, w, c) -> (c, h, w)
    image = np.transpose(image, (2,0,1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0

    return image

images = SegmentationData(df)

model.load_state_dict(torch.load('/home/antonio/workspace/ucl-thesis/best_model.pt'))

# create the output folder if it doesn't exist
output_dir = '/home/antonio/workspace/ucl-thesis/output_SPs'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for idx in range(len(images)):
    image = images[idx]

    logits_masks = model(image.to(DEVICE).unsqueeze(0))
    pred_mask = torch.sigmoid(logits_masks)
    pred_mask = (pred_mask > 0.5)*1.0

    # save the predicted mask in the output directory
    mask_name = f"mask_{idx}.jpg"
    output_path = os.path.join(output_dir, mask_name)
    torchvision.utils.save_image(pred_mask, output_path)

