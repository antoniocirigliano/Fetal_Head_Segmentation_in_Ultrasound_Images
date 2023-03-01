# Setup Visual Studio Code gpu runtime environment

#!pip install segmentation-models-pytorch
#!pip install -U git+https://github.com/albumentations-team/albumentations
#!pip install --upgrade opencv-contrib-python

# DGX

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.getcwd()

# Some Common Imports

import torch 
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Setup Configurations

img_path = "data/all_img"
mask_path = "data/all_masks"

img_files = [f for f in os.listdir(img_path) if f.endswith('.jpeg')]
mask_files = [f for f in os.listdir(mask_path) if f.endswith('.png')]

for f in img_files:
    image = cv2.imread(os.path.join(img_path, f))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(img_path, f.replace('.jpeg', '.png')), image)

img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
dataset = [(img_path + '/' + f, mask_path + '/' + f.replace('.png', '.png')) for f in img_files if f.replace('.png', '.png') in mask_files]

df = pd.DataFrame(dataset, columns=['images', 'masks'])
df.to_csv('file.csv', index=False)

CSV_FILE = 'file.csv'
DATA_DIR = '/home/antonio/workspace/ucl-thesis'

#DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

EPOCHS = 25
LR = 0.003
IMG_SIZE = 320 
BATCH_SIZE = 16

Encoder = 'timm-efficientnet-b0'
weights = 'imagenet'

#df = pd.read_csv(CSV_FILE)
#f.head()

#row = df.iloc[12]

#image_path = row.images
#mask_path = row.masks

#image = cv2.imread(image_path)

#mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0

#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        
#ax1.set_title('IMAGE')
#ax1.imshow(image)

#ax2.set_title('GROUND TRUTH')
#ax2.imshow(mask)

train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)

# Augmentation Functions

import albumentations as A

def get_train_augs():
  return A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE),
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.5)
  ])

def get_valid_augs():
  return A.Compose([
      A.Resize(IMG_SIZE, IMG_SIZE),
      
  ])

# Create Custom Dataset

from torch.utils.data import Dataset

class SegmentationDataset(Dataset):

  def __init__(self, df, augmentations):
    self.df = df
    self.augmentations = augmentations
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    row = self.df.iloc[idx]

    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #(h, w, c)
    mask= np.expand_dims(mask, axis = -1)

    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']
    
    #(h, w, c) -> (c, h, w)

    image = np.transpose(image, (2,0,1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask) / 255.0)

    return image, mask


trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_train_augs())

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")

import Plot1

#idx = 8

#image, mask = trainset[idx]
#show_image(image, mask)

# Load Dataset Into Batches

from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle = True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE)

print(f"total no. of batches in trainloader: {len(trainloader)}")
print(f"total no. of batches in validloader: {len(validloader)}")

for image, mask in trainloader:
  break
  
print(f"One batch image shape : {image.shape}")
print(f"One batch mask shape : {mask.shape}")

# Create Segmentation Model

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
model.to(DEVICE);

# Create Train and Validation Function

def train_fn(data_loader, model, optimizer):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader):

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  
  return total_loss / len(data_loader)

def eval_fn(data_loader, model):

  model.eval()
  total_loss = 0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader):
      
      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      logits, loss = model(images, masks)

      total_loss += loss.item()
  
  return total_loss / len(data_loader)

# Train Model

optimizer = torch.optim.Adam(model.parameters(), lr = LR)

# Initialize lists to store the loss values for the train set and the validation set
train_loss_list = []
val_loss_list = []

best_valid_loss = np.Inf

for i in range(EPOCHS):
  train_loss = train_fn(trainloader, model, optimizer)
  valid_loss = eval_fn(validloader, model)

  # Append the loss values to the lists
  train_loss_list.append(train_loss)
  val_loss_list.append(valid_loss)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    print("SAVED-MODEL")
    best_valid_loss = valid_loss

  print(f"Epoch _ {i+1} Train_loss : {train_loss} Valid_loss : {valid_loss}")

# Plot the loss for the train set and the validation set
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')

# Add labels and title to the plot
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Trend on Train and Validation Sets')

# Show the legend
plt.legend()

# Save the plot to a file
plt.savefig('loss_trend.png')

# Show the plot
plt.show()

# Inference

model.load_state_dict(torch.load('best_model.pt'))

for idx in range(len(validset)):
    image, mask = validset[idx]

    logits_masks = model(image.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    pred_mask = torch.sigmoid(logits_masks)
    pred_mask = (pred_mask > 0.5)*1.0

    show_image(image, mask, pred_mask.detach().cpu().squeeze(0))

# IOU

def iou(pred, target):
    """
    Compute the Intersection over Union (IoU) between the predicted segmentation and the ground truth segmentation.

    Arguments:
    - pred: a PyTorch tensor of shape (batch_size, H, W) representing the predicted segmentation
    - target: a PyTorch tensor of shape (batch_size, H, W) representing the ground truth segmentation

    Returns:
    - iou: a PyTorch tensor of shape (batch_size) representing the IoU between the predicted segmentation and the ground truth segmentation
    """
    # Move the tensors to the same device
    device = pred.device
    target = target.to(device)

    # Convert the tensors to binary arrays
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    # Compute the area of overlap
    pred = pred.to(device)
    intersection = (pred * target).sum((1, 2))

    # Compute the area of union
    union = (pred + target).sum((1, 2)) - intersection

    # Compute the IoU
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou


# Initialize a list to store the IoUs for each image
ious = []

for idx, (image, mask) in enumerate(validset):
    # Compute the predictions
    pred_mask = model(image.to(DEVICE).unsqueeze(0))

    # Compute the IoU for this image
    iou_value = iou(pred_mask.squeeze(0), mask)

    # Append the computed IoU to the list
    ious.append(iou_value)

# Calculate the average IoU across all images
avg_iou = torch.mean(torch.stack(ious))

# Print the result
print("The average IoU across all {} images in the validation set is {:.4f}".format(len(validset), avg_iou))