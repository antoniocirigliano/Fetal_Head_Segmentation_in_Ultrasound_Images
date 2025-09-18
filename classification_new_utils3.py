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
LR = 0.00003
IMG_SIZE = 320 
BATCH_SIZE = 8

Encoder = 'timm-efficientnet-b0'

def prepare_data(images_path, masks_path, labeled_filename, abdomen_femur_path, abdomen_femur_masks_path, abdome_femur_filename, images_black_border_path, masks_black_border_path, labeled_black_border_filename, unlabeled_filename, vol1_path, vol2_path, test_filename, vol1_path_test, vol2_path_test):
  
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

  # Aggiungi la colonna 'labels' con tutti i valori impostati a 1
  df_labeled['labels'] = 1

  df_labeled.to_csv(labeled_filename, index=False)

  # Abdomen and femur
  img_path = abdomen_femur_path
  mask_path = abdomen_femur_masks_path

  img_files = [f for f in os.listdir(img_path) if f.endswith('.jpeg')]
  mask_files = [f for f in os.listdir(mask_path) if f.endswith('.jpeg')]
  
  for f in img_files:
    image = cv2.imread(os.path.join(img_path, f))
    cv2.imwrite(os.path.join(img_path, f.replace('.jpeg', '.png')), image)

  for f in mask_files:
    mask = cv2.imread(os.path.join(mask_path, f))
    cv2.imwrite(os.path.join(mask_path, f.replace('.jpeg', '.png')), mask)

  img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
  mask_files = [f for f in os.listdir(mask_path) if f.endswith('.png')]
  dataset = [(img_path + '/' + f, mask_path + '/' + f.replace('.png', '.png')) for f in img_files if f.replace('.png', '.png') in mask_files]
  df_labeled = pd.DataFrame(dataset, columns=['images', 'masks'])

  # Aggiungi la colonna 'labels' con tutti i valori impostati a 0
  df_labeled['labels'] = 0

  df_labeled.to_csv(abdome_femur_filename, index=False)

  # Labeled images black border
  img_path = images_black_border_path
  mask_path =  masks_black_border_path

  img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
  mask_files = [f for f in os.listdir(mask_path) if f.endswith('.png')]
  dataset = [(os.path.join(img_path, f), os.path.join(mask_path, f)) for f in img_files if f in mask_files]
  df_labeled = pd.DataFrame(dataset, columns=['images', 'masks'])

  # Aggiungi la colonna 'labels' con tutti i valori impostati a 1
  df_labeled['labels'] = 1

  df_labeled.to_csv(labeled_black_border_filename, index=False)
  
  # Labeled images
  labels = labeled_filename
  df_labeled = pd.read_csv(labels)
  train_valid_head_df, test_head_df = train_test_split(df_labeled, test_size = 0.2, random_state = 42)
  train_head_df, valid_head_df = train_test_split(train_valid_head_df, test_size = 0.2, random_state = 42)
  
  # Abdomen and femur
  abdomen_femur = abdome_femur_filename
  df_abdomen_femur = pd.read_csv(abdomen_femur)
  train_valid_abdomen_femur_df, test_abdomen_femur_df = train_test_split(df_abdomen_femur, test_size = 0.2, random_state = 42)

  # Estrai casualmente lo 0%, 25%, 50%, 75% e 100% delle immagini dal DataFrame
  percentuale_da_estrazione = 0.25
  df_abdomen_femur_extract = train_valid_abdomen_femur_df.sample(frac=percentuale_da_estrazione, random_state=42)
  train_abdomen_femur_df, valid_abdomen_femur_df = train_test_split(df_abdomen_femur_extract, test_size = 0.2, random_state = 42)

  # Labeled images black border
  labels = labeled_black_border_filename
  df_labeled = pd.read_csv(labels)
  train_valid_black_border_df, test_black_border_df = train_test_split(df_labeled, test_size = 0.2, random_state = 100)
  train_black_border_df, valid_black_border_df = train_test_split(train_valid_black_border_df, test_size = 0.2, random_state = 100)
  
  # Unione dei dataframe di addestramento
  train_df = pd.concat([train_head_df, train_abdomen_femur_df, train_black_border_df], ignore_index=True)
  # train_df = pd.concat([train_head_df, train_black_border_df], ignore_index=True)

  # Unione dei dataframe di validazione
  valid_df = pd.concat([valid_head_df, valid_abdomen_femur_df, valid_black_border_df], ignore_index=True)
  # valid_df = pd.concat([valid_head_df, valid_black_border_df], ignore_index=True)

  # Unione dei dataframe di test
  test_df = pd.concat([test_head_df, test_abdomen_femur_df, test_black_border_df], ignore_index=True)

  # Mescola i DataFrame di addestramento
  train_df = train_df.sample(frac=1, random_state=42)

  # Mescola i DataFrame di validazione
  valid_df = valid_df.sample(frac=1, random_state=42)

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
  
  # Seleziona le prime righe dal DataFrame mescolato per il valid_df
  valid_unlabeled_df = df.head(53)

  # Rimuovi le righe selezionate dal DataFrame mescolato per ottenere il train_df
  train_unlabeled_df = df[~df.index.isin(valid_unlabeled_df.index)]
  
  # Labeled images: train set, validation set and test set
  train_set = SegmentationDataset(train_df, get_train_augs())
  valid_set = SegmentationDataset(valid_df, get_valid_augs())
  test_set = SegmentationDataset(test_df, get_valid_augs())
  
  print(f"Size of labeled train set : {len(train_set)}")
  print(f"Size of labeled valid set : {len(valid_set)}")
  print(f"Size of labeled test set : {len(test_set)}")
  
  # Test: mi serve per IoU della testa, testa col bordo nero e le non teste (addome e femore)
  test_head_set = SegmentationDataset(test_head_df, get_valid_augs())
  test_abdomen_femur_set = SegmentationDataset(test_abdomen_femur_df, get_valid_augs())
  test_black_border_set = SegmentationDataset(test_black_border_df, get_valid_augs())

  print(f"Size of head test set : {len(test_head_set)}")
  print(f"Size of head black border test set : {len(test_black_border_set)}")
  print(f"Size of abdomen and femur test set : {len(test_abdomen_femur_set)}")
  
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

  files1 = os.listdir(folder1)
  files2 = os.listdir(folder2)

  def get_file_number(filename):
    return int(re.findall(r'\d+', filename)[0])

  files1.sort(key=get_file_number)
  files2.sort(key=get_file_number)
  path_tuples = [(os.path.join(folder1, f1), os.path.join(folder2, f2)) for f1, f2 in zip(files1, files2)]

  with open(test_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['folder1', 'folder2'])
    writer.writerows(path_tuples)

  unlabeled_data = test_filename
  test_unlabeled_df = pd.read_csv(unlabeled_data)
  test_unlabeled_set = SegmentationDatasetCustom(test_unlabeled_df)

  print(f"Size of unlabeled test set : {len(test_unlabeled_set)}")

  return train_df, train_set, train_loader, valid_set, valid_loader, test_set, train_unlabeled_df, train_unlabeled_set, train_unlabeled_loader, valid_unlabeled_set, valid_unlabeled_loader, test_unlabeled_set, test_head_set, test_abdomen_femur_set, test_black_border_set 

def select_random_images(df, num_images):
  
  selected_images = df.sample(num_images, replace=False)
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
    label = row.labels

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

    # Convert label to float32
    label = torch.tensor(label, dtype=torch.float32)

    return image, mask, label

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

def binary_classification_metrics(predictions, targets):
    
    """
    Calcola precision, recall e accuracy per la classificazione binaria.

    Args:
        predictions (torch.Tensor): Tensor contenente le previsioni del modello.
        targets (torch.Tensor): Tensor contenente le etichette di classe.

    Returns:
        float: Precision
        float: Recall
        float: Accuracy
    """
    threshold = 0.5  # Soglia per convertire le previsioni in etichette binarie

    binary_predictions = (predictions > threshold).float()

    # print(binary_predictions)
    # print(binary_predictions.shape)

    true_positives = torch.sum(binary_predictions * targets)
    false_positives = torch.sum(binary_predictions * (1 - targets))
    false_negatives = torch.sum((1 - binary_predictions) * targets)
    true_negatives = torch.sum((1 - binary_predictions) * (1 - targets))

    # print(true_positives)
    # print(true_negatives)
    # print(false_negatives)
    # print(false_positives)

    precision_pos = true_positives / (true_positives + false_positives + 1e-10)
    recall_pos = true_positives / (true_positives + false_negatives + 1e-10)

    precision_neg = true_negatives / (true_negatives + false_negatives + 1e-10)
    recall_neg = true_negatives / (true_negatives + false_positives + 1e-10)

    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives + 1e-10)

    # print(precision)
    # print(recall)
    # print(accuracy)

    return precision_pos.item(), recall_pos.item(), precision_neg.item(), recall_neg.item(), accuracy.item()
