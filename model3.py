from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils3 import *

class SegmentationModel(nn.Module):

  def __init__(self):
    super(SegmentationModel, self).__init__()

    self.arc=sample_data.Unet(
      encoder_name = Encoder,
      encoder_weights = 'imagenet',
      in_channels = 3, 
      classes = 1,
      activation = None
    )

  def forward(self, images=None, masks=None, unlabeled_images1=None, unlabeled_images2=None):
    
    if images is not None:
      logits = self.arc(images)
      if masks is not None:
        loss1 = DiceLoss(mode='binary')(logits, masks)
        loss2 = nn.BCEWithLogitsLoss()(logits, masks)
        labeled_loss = (loss1 + loss2) / 2
      else:
        labeled_loss = None
    else:
      logits = None
      labeled_loss = None
    
    if unlabeled_images1 is not None and unlabeled_images2 is not None:
      unlabeled_logits1 = self.arc(unlabeled_images1)
      unlabeled_logits2 = self.arc(unlabeled_images2)
      unlabeled_probs1 = torch.sigmoid(unlabeled_logits1)
      unlabeled_probs2 = torch.sigmoid(unlabeled_logits2)
      #print(unlabeled_probs1.shape)
      unlabeled_loss = nn.MSELoss()(unlabeled_probs1, unlabeled_probs2)
    else:
      unlabeled_loss = None

    #print(logits.shape)
    #print(labeled_loss, unlabeled_loss)

    return logits, labeled_loss, unlabeled_loss