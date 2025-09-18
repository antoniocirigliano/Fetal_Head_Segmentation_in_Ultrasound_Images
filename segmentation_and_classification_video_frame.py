from classification_new_main3 import *
from classification_new_model3 import *
from classification_new_utils3 import *

import os
import warnings

import pandas as pd
import cv2

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.getcwd()

def save_image(image, pred_image, save_path_template, index):
    save_path = save_path_template.format(index)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    
    axs[0].set_title('IMAGE')
    axs[0].imshow(image.permute(1,2,0).squeeze(), cmap='gray')
        
    axs[1].set_title('MODEL OUTPUT')
    axs[1].imshow(pred_image.permute(1,2,0).squeeze(), cmap='gray')
    
    plt.subplots_adjust(wspace=0.2, hspace=0)
    plt.savefig(save_path)
    plt.close()

class SegmentationDatasetVideoFrames(Dataset):

  def __init__(self, f):
    self.f = f
  
  def __len__(self):
    return len(self.f)

  def __getitem__(self, idx):
    
    row = self.f.iloc[idx]

    image_path = row.images

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (320, 320))
     
    # (h, w, c) -> (c, h, w)
    image = np.transpose(image, (2,0,1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0

    return image

class SegmentationDatasetVideoFrames1(Dataset):

  def __init__(self, f):
    self.f = f
  
  def __len__(self):
    return len(self.f)

  def __getitem__(self, idx):
    
    row = self.f.iloc[idx]

    image_path = row.images

    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
    # (h, w, c) -> (c, h, w)
    image = np.transpose(image, (2,0,1)).astype(np.float32)
    image = torch.Tensor(image) / 255.0

    return image

img_path = "/home/antonio/workspace/ucl-thesis/video_frames/Operator_6"

output_folder = "/home/antonio/workspace/ucl-thesis/classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler/segmentation_and_classification_with_progress_bar_Operator_6"

img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
#img_files.sort()
img_files.sort(key=lambda x: int(x.split(".")[0]))

dataset = [(os.path.join(img_path, f)) for f in img_files]

df_images = pd.DataFrame(dataset, columns=['images'])
df_images.to_csv('/home/antonio/workspace/ucl-thesis/csv/Operator_6.csv', index=False)

df_images = pd.read_csv('/home/antonio/workspace/ucl-thesis/csv/Operator_6.csv')
data_set = SegmentationDatasetVideoFrames(df_images)

# Load model
model = SegmentationModel().to(DEVICE)
model.load_state_dict(torch.load('classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler.pt'))

progress_colors = []

# model.eval()

# with torch.no_grad():

#   # Inizializza una lista per contenere i colori delle barre di avanzamento
#   progress_colors = []

#   for idx in range(len(data_set)):

#     image = data_set[idx]

#     logits_masks, _, _, class_probs, _ = model(image.to(DEVICE).unsqueeze(0), None, None, None, None) #(C, H, W) -> (1, C, H, W)
    
#     pred_mask = torch.sigmoid(logits_masks)
#     pred_mask = (pred_mask > 0.5)*1.0

#     # Applica una soglia alle probabilità per ottenere le etichette binarie
#     binary_predictions = (class_probs > 0.5).int().item()  # Soglia 0.5

#     if binary_predictions == 0:
#         # Se la predizione è 0, imposta la maschera a zero
#         pred_mask.fill_(0)

#     save_image(image, pred_mask.detach().cpu().squeeze(0), '/home/antonio/workspace/ucl-thesis/classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler/Operator_6/image{}.png', idx)

img_path_1 = "/home/antonio/workspace/ucl-thesis/classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler/Operator_6"

img_files = [f for f in os.listdir(img_path_1) if f.endswith('.png')]
#img_files.sort()
img_files.sort(key=lambda x: int(x.split("image")[1].split(".")[0]))

dataset = [(os.path.join(img_path_1, f)) for f in img_files]

df_images = pd.DataFrame(dataset, columns=['images'])

data_set_1 = SegmentationDatasetVideoFrames1(df_images)

model.eval()

with torch.no_grad():

  # Stampa le etichette predette sull'immagine e salva l'immagine modificata
  for idx in range(len(data_set)):
    
    image = data_set[idx]
    image1 = data_set_1[idx]

    _, _, _, class_probs, _ = model(image.to(DEVICE).unsqueeze(0), None, None, None, None)  # (C, H, W) -> (1, C, H, W)
        
    # Applica una soglia alle probabilità per ottenere le etichette binarie
    binary_predictions = (class_probs > 0.5).int().item()  # Soglia 0.5
        
    # Aggiungi il colore della barra di avanzamento corrispondente alla predizione
    progress_colors.append('green' if binary_predictions else 'red')
        
    # Disegna le barre di avanzamento sopra l'immagine
    plt.imshow(image1.numpy().transpose(1, 2, 0))

    # Disegna la barra di avanzamento in alto sopra l'immagine
    for i, color in enumerate(progress_colors):
      plt.axvline(x=i * image1.shape[2] / len(data_set_1), color=color, linewidth= image1.shape[2] / len(data_set_1), alpha=0.5, ymax=0.05)
    plt.axis('off')

    # Aggiorna la visualizzazione della figura
    plt.draw()

    # Salva l'immagine modificata nella cartella specifica
    output_path = os.path.join(output_folder, f"{idx}.png")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()