import os
import sys
import torch 
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import csv 
import re
import albumentations as A
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import segmentation_models_pytorch as sample_data
from segmentation_models_pytorch.losses import DiceLoss
import random

if torch.cuda.is_available():
  DEVICE = 'cuda'
else:
  DEVICE = 'cpu'

seed = 4

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    '''
    _init_fn sets the seed for each worker thread in a multi-thread environment.
    The seed is set to the value of the "seed" variable casted as an integer.
    '''
    np.random.seed(int(seed))

EPOCHS = 150
LR = 0.008
IMG_SIZE = 320 
BATCH_SIZE = 8

Encoder = 'timm-efficientnet-b0'

def prepare_data(images_path, masks_path, labeled_filename, unlabeled_filename, vol1_path, vol2_path, test_filename, vol1_path_test, vol2_path_test): #vol3_path_test, vol4_path_test
  
  # Labeled images
  img_path = images_path
  mask_path = masks_path

  img_files = [f for f in os.listdir(img_path) if f.endswith('.jpeg')]
  mask_files = [f for f in os.listdir(mask_path) if f.endswith('.png')]
  
  for f in img_files:
    image = cv2.imread(os.path.join(img_path, f))
    cv2.imwrite(os.path.join(img_path, f.replace('.jpeg', '.png')), image)


  img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
  dataset = [(img_path + '/' + f, mask_path + '/' + f.replace('.png', '.png')) for f in img_files if f.replace('.png', '.png') in mask_files]

  df_labeled = pd.DataFrame(dataset, columns=['images', 'masks'])
  df_labeled.to_csv(labeled_filename, index=False)

  labels = labeled_filename
  df_labeled = pd.read_csv(labels)
  train_valid_df, test_df = train_test_split(df_labeled, test_size = 0.2, random_state = 42)
  train_df, valid_df = train_test_split(train_valid_df, test_size = 0.2, random_state = 42)
  
  # Unlabeled images
  folder1 = vol1_path
  folder2 = vol2_path

  files1 = os.listdir(folder1)
  files2 = os.listdir(folder2)

  def get_file_number(filename):
    return int(re.findall(r'\d+', filename)[0])

  files1.sort(key=get_file_number)
  files2.sort(key=get_file_number)

  path_tuples = [(os.path.join(folder1, f1), os.path.join(folder2, f2)) for f1, f2 in zip(files1, files2)]

  with open(unlabeled_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['folder1', 'folder2'])
    writer.writerows(path_tuples)

  unlabeled_data = unlabeled_filename
  df = pd.read_csv(unlabeled_data)

  # Mescola casualmente le righe del DataFrame
  df = df.sample(frac=1, random_state=42)

  # Seleziona le prime 22 righe dal DataFrame mescolato per il valid_df
  valid_unlabeled_df = df.head(22)

  # # Stampa le prime 22 righe
  # print("Prime 22 righe del DataFrame:")
  # print(valid_unlabeled_df)

  # Rimuovi le righe selezionate dal DataFrame mescolato per ottenere il train_df
  train_unlabeled_df = df[~df.index.isin(valid_unlabeled_df.index)]

  # # Stampa le righe rimosse
  # print("\nRighe rimosse per creare train_unlabeled_df:")
  # print(df[df.index.isin(valid_unlabeled_df.index)])
  
  # Labeled images: train set, validation set and test set
  train_set = SegmentationDataset(train_df, get_train_augs())
  valid_set = SegmentationDataset(valid_df, get_valid_augs())
  test_set = SegmentationDataset(test_df, get_valid_augs())
  
  print(f"Size of labeled train set : {len(train_set)}")
  print(f"Size of labeled valid set : {len(valid_set)}")
  print(f"Size of labeled test set : {len(test_set)}")
  
  # Unlabeled images: train set and validation set
  train_unlabeled_set = SegmentationDatasetCustom(train_unlabeled_df)
  valid_unlabeled_set = SegmentationDatasetCustom(valid_unlabeled_df)

  print(f"Size of unlabeled train set : {len(train_unlabeled_set)}")
  print(f"Size of unlabeled valid set : {len(valid_unlabeled_set)}")
  
  # Labeled and unlabeled images: train and validation loader
  train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle = True)
  valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
  
  train_unlabeled_loader = DataLoader(train_unlabeled_set, batch_size=BATCH_SIZE, shuffle = True)
  valid_unlabeled_loader = DataLoader(valid_unlabeled_set, batch_size=BATCH_SIZE)
  
  print(f"total no. of batches in train labeled loader: {len(train_loader)}")
  print(f"total no. of batches in valid labeled loader: {len(valid_loader)}")
  
  print(f"total no. of batches in train unlabeled loader: {len(train_unlabeled_loader)}")
  print(f"total no. of batches in valid unlabeled loader: {len(valid_unlabeled_loader)}")

  # Prepare images for test: 24w and 25w
  folder1 = vol1_path_test
  folder2 = vol2_path_test
  # folder3 = vol3_path_test
  # folder4 = vol4_path_test

  files1 = os.listdir(folder1)
  files2 = os.listdir(folder2)
  # files3 = os.listdir(folder3)
  # files4 = os.listdir(folder4)

  def get_file_number(filename):
    return int(re.findall(r'\d+', filename)[0])

  files1.sort(key=get_file_number)
  files2.sort(key=get_file_number)
  # files3.sort(key=get_file_number)
  # files4.sort(key=get_file_number)

  path_tuples = [(os.path.join(folder1, f1), os.path.join(folder2, f2)) for f1, f2 in zip(files1, files2)]

  with open(test_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['folder1', 'folder2'])
    writer.writerows(path_tuples)

  unlabeled_data = test_filename
  test_unlabeled_df = pd.read_csv(unlabeled_data)

  test_unlabeled_set = SegmentationDatasetCustom(test_unlabeled_df)

  print(f"Size of unlabeled test set : {len(test_unlabeled_set)}")

  return train_df, train_set, train_loader, valid_set, valid_loader, test_set, train_unlabeled_df, train_unlabeled_set, train_unlabeled_loader, valid_unlabeled_set, valid_unlabeled_loader, test_unlabeled_set

def select_random_images(df, num_images):
  
  # qui avrei dovuto settare il seed in modo tale da rendere l'esperimento riproducibile (Fatto!)

  selected_images = df.sample(num_images, replace=False)
  # print("Indici delle immagini estratte:")
  # print(selected_images.index.tolist())  # Stampare gli indici delle immagini estratte
  data_set = SegmentationDatasetCustom(selected_images)
  data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle = True)

  return data_loader

def get_train_augs():
  return A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p = 0.5),
    A.VerticalFlip(p = 0.4),
    A.Rotate(limit=45, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.1),
    A.ElasticTransform(alpha=5, sigma=7, alpha_affine=5, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.2),
  ])

def get_valid_augs():
  return A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),    
  ])

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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask= np.expand_dims(mask, axis = -1)

    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']
    
    # (h, w, c) -> (c, h, w)
    image = np.transpose(image, (2,0,1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask) / 255.0)

    return image, mask

class SegmentationDatasetCustom(Dataset):

  def __init__(self, f):
    self.f = f
  
  def __len__(self):
    return len(self.f)

  def __getitem__(self, idx):
    
    row = self.f.iloc[idx]

    image1_path = row.folder1
    image2_path = row.folder2

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    image1 = cv2.resize(image1, (IMG_SIZE, IMG_SIZE))
    image2 = cv2.resize(image2, (IMG_SIZE, IMG_SIZE))
    
    # (h, w, c) -> (c, h, w)
    image1 = np.transpose(image1, (2,0,1)).astype(np.float32)
    image1 = torch.Tensor(image1) / 255.0

    image2 = np.transpose(image2, (2,0,1)).astype(np.float32)
    image2 = torch.Tensor(image2) / 255.0

    return image1, image2

class SegmentationDatasetCustomTest(Dataset):

  def __init__(self, f):
    self.f = f
  
  def __len__(self):
    return len(self.f)

  def __getitem__(self, idx):
    
    row = self.f.iloc[idx]

    image1_path = row.folder1
    image2_path = row.folder2
    image3_path = row.folder3
    image4_path = row.folder4

    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    image3 = cv2.imread(image3_path)
    image4 = cv2.imread(image4_path)

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)

    image1 = cv2.resize(image1, (IMG_SIZE, IMG_SIZE))
    image2 = cv2.resize(image2, (IMG_SIZE, IMG_SIZE))
    image3 = cv2.resize(image3, (IMG_SIZE, IMG_SIZE))
    image4 = cv2.resize(image4, (IMG_SIZE, IMG_SIZE))
    
    # (h, w, c) -> (c, h, w)
    image1 = np.transpose(image1, (2,0,1)).astype(np.float32)
    image1 = torch.Tensor(image1) / 255.0

    image2 = np.transpose(image2, (2,0,1)).astype(np.float32)
    image2 = torch.Tensor(image2) / 255.0
  
    image3 = np.transpose(image3, (2,0,1)).astype(np.float32)
    image3 = torch.Tensor(image3) / 255.0

    image4 = np.transpose(image4, (2,0,1)).astype(np.float32)
    image4 = torch.Tensor(image4) / 255.0

    return image1, image2, image3, image4

def iou(image1, image2, type):

  """
  Compute the Intersection over Union (IoU) between the predicted segmentation and the ground truth segmentation.

  Arguments:
  - pred: a PyTorch tensor of shape (batch_size, H, W) representing the predicted segmentation.
  - target: a PyTorch tensor of shape (batch_size, H, W) representing the ground truth segmentation.

  Returns:
  - iou: a PyTorch tensor of shape (batch_size) representing the IoU between the predicted segmentation and the ground truth segmentation.
  """

  if type == 'labeled':
    
    # Convert the logits to probabilities using a sigmoid activation function
    image1 = torch.sigmoid(image1)

    # Convert the tensors to binary arrays
    image1 = (image1 > 0.5).float()
    image2 = image2.float()

  else:
    
    # Convert the logits to probabilities using a sigmoid activation function
    image1 = torch.sigmoid(image1)
    image2 = torch.sigmoid(image2)

    # Convert the tensors to binary arrays
    image1 = (image1 > 0.5).float()
    image2 = (image2 > 0.5).float()

  # Compute the area of overlap
  intersection = (image1 * image2).sum((1, 2))

  # Compute the area of union
  union = (image1 + image2).sum((1, 2)) - intersection

  # Compute the IoU
  iou = (intersection + 1e-6) / (union + 1e-6)

  return iou
