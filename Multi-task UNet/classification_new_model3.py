from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

from classification_new_utils3 import *

class SegmentationModel(nn.Module):

  def __init__(self):
    super(SegmentationModel, self).__init__()

    aux_params=dict(
    pooling='avg',             # one of 'avg', 'max'
    dropout=0.5,               # dropout ratio, default is None
    activation='sigmoid',      # activation function, default is None
    classes=1,                 # define number of output labels
    )
    
    self.arc=sample_data.Unet(
      encoder_name = Encoder,
      encoder_weights = 'imagenet',
      in_channels = 3, 
      classes = 1,
      activation = None,
      aux_params = aux_params
    )

  def forward(self, images=None, masks=None, unlabeled_images1=None, unlabeled_images2=None, labels=None):
    
    # print(labels.shape)
    # print(labels)

    # Labeled loss
    if images is not None:
      logits, _ = self.arc(images)
      if masks is not None:
        loss1 = DiceLoss(mode='binary')(logits, masks)
        loss2 = nn.BCEWithLogitsLoss()(logits, masks)
        labeled_loss = (loss1 + loss2) / 2
      else:
        labeled_loss = None
    else:
      logits = None
      labeled_loss = None
    
    # Unlabeled loss
    if unlabeled_images1 is not None and unlabeled_images2 is not None:
      unlabeled_logits1, _ = self.arc(unlabeled_images1)
      unlabeled_logits2, _ = self.arc(unlabeled_images2)
      unlabeled_probs1 = torch.sigmoid(unlabeled_logits1)
      unlabeled_probs2 = torch.sigmoid(unlabeled_logits2)
      unlabeled_loss = nn.MSELoss()(unlabeled_probs1, unlabeled_probs2)
    else:
      unlabeled_loss = None

    # Classificazione
    if images is not None:
      _, class_probs = self.arc(images)
      # print(class_probs.shape)
      # print(class_probs)
      if labels is not None:
        # Utilizzare squeeze per rimuovere la dimensione 1
        class_probs = class_probs.squeeze(dim=1)
        
        # print(class_probs.shape)
        # print(class_probs)
      
        classification_loss = nn.BCELoss()(class_probs, labels)
        #print(classification_loss)
      else:
        classification_loss = None
    else:
      class_probs = None
      classification_loss = None
    
    return logits, labeled_loss, unlabeled_loss, class_probs, classification_loss
