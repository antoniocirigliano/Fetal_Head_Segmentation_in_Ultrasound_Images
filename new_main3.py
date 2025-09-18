from __future__ import print_function, division, absolute_import

from torch.utils.tensorboard import SummaryWriter

from itertools import zip_longest
import time

from new_utils3 import *
from new_model3 import *

def train(train_loader, valid_loader, unlabeled_train_df, unlabeled_valid_loader, alpha, input_model, output_model, figpath):

  model = SegmentationModel().to(DEVICE)

  if input_model is not None:
    model.load_state_dict(torch.load(input_model))

  optimizer = optim.Adam(model.parameters(), lr = LR)
  #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
  #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99)

  # -------------------------------------------------------------------------

  def iter_dataloader(data_loader, unlabeled_loader, model, training):

    total_loss = 0.0

    for (labeled_images, masks), (unlabeled_images1, unlabeled_images2) in tqdm(zip(data_loader, unlabeled_loader),  total=len(unlabeled_loader)):

      labeled_images = labeled_images.to(DEVICE) if labeled_images is not None else None
      masks = masks.to(DEVICE) if masks is not None else None

      unlabeled_images1 = unlabeled_images1.to(DEVICE) if unlabeled_images1 is not None and alpha is not 0 else None
      unlabeled_images2 = unlabeled_images2.to(DEVICE) if unlabeled_images2 is not None and alpha is not 0 else None

      if training == True:
        optimizer.zero_grad()

      _, labeled_loss, unlabeled_loss = model(labeled_images, masks, unlabeled_images1, unlabeled_images2)

      loss = 0.0
      if labeled_loss is not None:
        loss += labeled_loss

      if unlabeled_loss is not None:
        loss += alpha * unlabeled_loss

      if labeled_loss is not None or unlabeled_loss is not None:
        total_loss += loss.item() 

      if training == True and (labeled_loss is not None or unlabeled_loss is not None):
        loss.backward()
        optimizer.step()
    
    avg_loss = total_loss / (len(data_loader) + alpha * len(unlabeled_loader))

    return avg_loss

  # -------------------------------------------------------------------------

  def train_fn(model, train_loader, unlabeled_train_loader):
    
    model.train()

    loss = iter_dataloader(train_loader, unlabeled_train_loader, model, training=True)

    return loss

  # -------------------------------------------------------------------------
  
  def eval_fn(model, valid_loader, unlabeled_valid_loader):
    
    model.eval()

    with torch.no_grad():
      loss = iter_dataloader(valid_loader, unlabeled_valid_loader, model, training=False)

    return loss

  # -------------------------------------------------------------------------
  
  print('Training\n')

  start = time.time()

  train_loss_list = []
  valid_loss_list = []

  # Svuoto la lista dalle loss del caso precedente
  train_loss_list.clear()
  valid_loss_list.clear()

  best_valid_loss = np.Inf

  writer = SummaryWriter()

  for epoch in range(1, EPOCHS+1):
    
    unlabeled_train_loader = select_random_images(unlabeled_train_df, 172)
      
    train_loss = train_fn(model, train_loader, unlabeled_train_loader)
    valid_loss = eval_fn(model, valid_loader, unlabeled_valid_loader)

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/valid", valid_loss, epoch)

    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)

    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), output_model)
      print("Saved best model")
      print(f'Minimum Validation Loss of {best_valid_loss:.6f} at epoch {epoch}/{EPOCHS}\n')

    #scheduler.step()

    #print(f"Epoch [{epoch}/{EPOCHS}] Learning Rate: {scheduler.get_lr()}")
    print(f"Epoch [{epoch}/{EPOCHS}] Train Loss: [{train_loss:.4f}] Validation Loss: [{valid_loss:.4f}]")
    
    print('-------------------------------------------------------------------------------\n')

  writer.flush()
  writer.close()
  
  end = time.time()
  elapsed_time = (end - start) / 60

  print(f'>>> Training Complete: {elapsed_time:.2f} minutes\n')

  # -------------------------------------------------------------------------

  # Plot the loss for the train set and the validation set
  plt.plot(train_loss_list, label='Train Loss')
  plt.plot(valid_loss_list, label='Validation Loss')

  # Add labels and title to the plot
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Trend on Train and Validation Sets')

  # Show the legend
  plt.legend()

  # Save the plot to a file
  plt.savefig(figpath)

  return output_model

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def compute_metrics(data_set, type, model):

  ious = []

  model.eval()

  with torch.no_grad():

    if type == 'labeled':

      for idx, (image, mask) in enumerate(data_set):
        # Compute the predictions
        pred_mask, _, _ = model(images=image.to(DEVICE).unsqueeze(0))

        # Move the mask tensor to the same device as the pred_mask tensor
        mask = mask.to(pred_mask.device)
      
        # Compute the IoU for this image
        iou_value = iou(pred_mask.squeeze(0), mask, type)
      
        # Append the computed IoU to the list
        ious.append(iou_value) 
    
    else :
      
      for idx, (image1, image2) in enumerate(data_set):
        # Compute the predictions
        pred_mask1, _, _ = model(images=image1.to(DEVICE).unsqueeze(0))
        pred_mask2, _, _ = model(images=image2.to(DEVICE).unsqueeze(0))

        pred_mask2 = pred_mask2.to(pred_mask1.device)
      
        # Compute the IoU for this image
        iou_value = iou(pred_mask1.squeeze(0), pred_mask2.squeeze(0), type)
      
        # Append the computed IoU to the list
        ious.append(iou_value) 

    avg_iou = torch.mean(torch.stack(ious))
  
  ious.clear()

  if type == 'labeled':
    print(f"The average IoU across all {len(data_set)} images in the labeled set is {avg_iou}")
  else:
    print(f"The average IoU across all {len(data_set)} images in the unlabeled set is {avg_iou}")
          
  return avg_iou

def evaluate(train_set, valid_set, test_set, train_unlabeled_set, valid_unlabeled_set, test_unlabeled_set, test_head_set, test_abdomen_femur_set, test_black_border_set, saved_model):

  model = SegmentationModel().to(DEVICE)

  if saved_model is not None:
    model.load_state_dict(torch.load(saved_model))
    print(f'Loaded {saved_model} model')
  else:
    print('Error: model not loaded')
    return None
  
  train_iou = compute_metrics(train_set, 'labeled', model)
  train_unlabeled_iou = compute_metrics(train_unlabeled_set, 'unlabeled', model)

  print('>>> Evaluation on Train set completed')
  
  valid_iou = compute_metrics(valid_set, 'labeled', model)
  valid_unlabeled_iou = compute_metrics(valid_unlabeled_set, 'unlabeled', model)

  print('>>> Evaluation on Validation set completed')
  
  test_iou = compute_metrics(test_set, 'labeled', model)
  test_unlabeled_iou = compute_metrics(test_unlabeled_set, 'unlabeled', model)

  test_head_iou = compute_metrics(test_head_set, 'labeled', model)
  test_black_border_iou = compute_metrics(test_black_border_set, 'labeled', model)
  test_abdomen_femur_iou = compute_metrics(test_abdomen_femur_set, 'labeled', model)

  print('>>> Evaluation on Test set completed')

  print('\n>>> Evaluation completed\n')

  return train_iou, valid_iou, test_iou, train_unlabeled_iou, valid_unlabeled_iou, test_unlabeled_iou, test_head_iou, test_black_border_iou, test_abdomen_femur_iou