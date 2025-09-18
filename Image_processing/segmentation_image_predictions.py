from new_main3 import *
from new_model3 import *
from new_utils3 import *

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.getcwd()

# Per salvare gli output del modello
import matplotlib.pyplot as plt 

def save_image1(image, mask, pred_image, save_path_template, index):
    save_path = save_path_template.format(index)
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
    
    axs[0].set_title('IMAGE')
    axs[0].imshow(image.permute(1,2,0).squeeze(), cmap='gray')
        
    axs[1].set_title('GROUND TRUTH')
    axs[1].imshow(mask.permute(1,2,0).squeeze(), cmap='gray')
        
    axs[2].set_title('MODEL OUTPUT')
    axs[2].imshow(pred_image.permute(1,2,0).squeeze(), cmap='gray')
    
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(save_path)
    plt.close()

def save_image2(image1, pred_image1, image2, pred_image2, save_path_template, index):
    save_path = save_path_template.format(index)
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(10,5))
    
    axs[0].set_title('IMAGE')
    axs[0].imshow(image1.permute(1,2,0).squeeze(), cmap='gray')
        
    axs[1].set_title('MODEL OUTPUT')
    axs[1].imshow(pred_image1.permute(1,2,0).squeeze(), cmap='gray')
        
    axs[2].set_title('IMAGE')
    axs[2].imshow(image2.permute(1,2,0).squeeze(), cmap='gray')

    axs[3].set_title('MODEL OUTPUT')
    axs[3].imshow(pred_image2.permute(1,2,0).squeeze(), cmap='gray')
    
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(save_path)
    plt.close()

def save_image3(image, pred_image, save_path_template, index):
    save_path = save_path_template.format(index)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    axs[0].set_title('IMAGE')
    axs[0].imshow(image.permute(1,2,0).squeeze(), cmap='gray')
        
    axs[1].set_title('MODEL OUTPUT')
    axs[1].imshow(pred_image.permute(1,2,0).squeeze(), cmap='gray')
    
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(save_path)
    plt.close()

# # Labeled
# labeled_filename = r'csv/labeled_data.csv'
# df_labeled = pd.read_csv(labeled_filename)
# labeled_set = SegmentationDataset(df_labeled, get_valid_augs())

# # Abdomen femur
# abdomen_femur_filename = r'csv/labeled_abdome_femur_data.csv'
# df_abdomen_femur = pd.read_csv(abdomen_femur_filename)
# abdomen_femur_set = SegmentationDataset(df_abdomen_femur, get_valid_augs())

# # Black_border
# black_border_filename = r'csv/labeled_black_border_data.csv'
# df_black_border = pd.read_csv(black_border_filename)
# black_border_set = SegmentationDataset(df_black_border, get_valid_augs())

# Unlabeled 23w
unlabeled_filename = r'csv/unlabeled_data.csv'
df_unlabeled = pd.read_csv(unlabeled_filename)
df_unlabeled_random = df_unlabeled.sample(200, replace=False)
unlabeled_set = SegmentationDatasetCustom(df_unlabeled_random)

# # Unlabeled 24w and 25w
# unlabeled_filename = r'csv/unlabeled_test_data.csv'
# df_unlabeled = pd.read_csv(unlabeled_filename)
# df_unlabeled_random = df_unlabeled.sample(200, replace=False)
# unlabeled_test_set = SegmentationDatasetCustom(df_unlabeled_random)

# # Unlabeled 21w and 22w
# unlabeled_filename = r'csv/21w_22w.csv'
# df_unlabeled = pd.read_csv(unlabeled_filename)
# df_unlabeled_random = df_unlabeled.sample(200, replace=False)
# unlabeled_21w_22w_set = SegmentationDatasetCustom(df_unlabeled_random)

# # Unlabeled 21w and 24w
# unlabeled_filename = r'csv/21w_24w.csv'
# df_unlabeled = pd.read_csv(unlabeled_filename)
# df_unlabeled_random = df_unlabeled.sample(200, replace=False)
# unlabeled_21w_24w_set = SegmentationDatasetCustom(df_unlabeled_random)

# Load model
model = SegmentationModel().to(DEVICE)
model.load_state_dict(torch.load('chiara_new_initial_weights.pt'))

model.eval()

with torch.no_grad():

    # for idx in range(len(labeled_set)):
    #     image, mask = labeled_set[idx]
    #     logits_masks, _ = model(images=image.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     pred_mask = torch.sigmoid(logits_masks)
    #     pred_mask = (pred_mask > 0.5)*1.0
    #     save_image1(image, mask, pred_mask.detach().cpu().squeeze(0), '/home/antonio/workspace/ucl-thesis/chiara-initial_weights/labeled/image{}.png', idx)
    
    # for idx in range(len(abdomen_femur_set)):
    #     image, mask = abdomen_femur_set[idx]
    #     logits_masks, _ = model(images=image.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     pred_mask = torch.sigmoid(logits_masks)
    #     pred_mask = (pred_mask > 0.5)*1.0
    #     save_image1(image, mask, pred_mask.detach().cpu().squeeze(0), '/home/antonio/workspace/ucl-thesis/chiara-initial_weights/abdomen_femur/image{}.png', idx)
    
    # for idx in range(len(black_border_set)):
    #     image, mask = black_border_set[idx]
    #     logits_masks, _, = model(images=image.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     pred_mask = torch.sigmoid(logits_masks)
    #     pred_mask = (pred_mask > 0.5)*1.0
    #     save_image1(image, mask, pred_mask.detach().cpu().squeeze(0), '/home/antonio/workspace/ucl-thesis/chiara-initial_weights/black_border/image{}.png', idx)
    
    for idx in range(len(unlabeled_set)):
        image1, image2 = unlabeled_set[idx]
        logits_masks1, _, _ = model(images=image1.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
        logits_masks2, _, _ = model(images=image2.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
        pred_mask1 = torch.sigmoid(logits_masks1)
        pred_mask1 = (pred_mask1 > 0.5)*1.0
        pred_mask2 = torch.sigmoid(logits_masks2)
        pred_mask2 = (pred_mask2 > 0.5)*1.0
        save_image2(image1, pred_mask1.detach().cpu().squeeze(0), image2, pred_mask2.detach().cpu().squeeze(0), '/home/antonio/workspace/ucl-thesis/chiara_new_initial_weights/23w/image{}.png', idx)
    
    # for idx in range(len(unlabeled_test_set)):
    #     image1, image2 = unlabeled_test_set[idx]
    #     logits_masks1, _, _ = model(images=image1.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     logits_masks2, _, _ = model(images=image2.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     pred_mask1 = torch.sigmoid(logits_masks1)
    #     pred_mask1 = (pred_mask1 > 0.5)*1.0
    #     pred_mask2 = torch.sigmoid(logits_masks2)
    #     pred_mask2 = (pred_mask2 > 0.5)*1.0
    #     save_image2(image1, pred_mask1.detach().cpu().squeeze(0), image2, pred_mask2.detach().cpu().squeeze(0), '/home/antonio/workspace/ucl-thesis/inizial_weights/24w_25w/image{}.png', idx)

    # for idx in range(len(unlabeled_21w_22w_set)):
    #     image1, image2 = unlabeled_21w_22w_set[idx]
    #     logits_masks1, _, _ = model(images=image1.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     logits_masks2, _, _ = model(images=image2.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     pred_mask1 = torch.sigmoid(logits_masks1)
    #     pred_mask1 = (pred_mask1 > 0.5)*1.0
    #     pred_mask2 = torch.sigmoid(logits_masks2)
    #     pred_mask2 = (pred_mask2 > 0.5)*1.0
    #     save_image2(image1, pred_mask1.detach().cpu().squeeze(0), image2, pred_mask2.detach().cpu().squeeze(0), '/home/antonio/workspace/ucl-thesis/case-3_alpha-0.5_epochs-150_lr-0.008_batch-size-8_gamma-0.5_step-size-50/21w_22w/image{}.png', idx)

    # for idx in range(len(unlabeled_21w_24w_set)):
    #     image1, image2 = unlabeled_21w_24w_set[idx]
    #     logits_masks1, _, _ = model(images=image1.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     logits_masks2, _, _ = model(images=image2.to(DEVICE).unsqueeze(0)) #(C, H, W) -> (1, C, H, W)
    #     pred_mask1 = torch.sigmoid(logits_masks1)
    #     pred_mask1 = (pred_mask1 > 0.5)*1.0
    #     pred_mask2 = torch.sigmoid(logits_masks2)
    #     pred_mask2 = (pred_mask2 > 0.5)*1.0
    #     save_image2(image1, pred_mask1.detach().cpu().squeeze(0), image2, pred_mask2.detach().cpu().squeeze(0), '/home/antonio/workspace/ucl-thesis/case-3_alpha-0.5_epochs-150_lr-0.008_batch-size-8_gamma-0.5_step-size-50/21w_24w/image{}.png', idx)
