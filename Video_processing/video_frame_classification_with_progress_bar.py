import torch
from torchvision import transforms
from PIL import Image
import os
import warnings
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from classification_new_model3 import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.getcwd()

img_path = "/home/antonio/workspace/ucl-thesis/video_frames/Patient017-Operator017-Novice-20230519-145030"

img_files = [f for f in os.listdir(img_path) if f.endswith('.png')]

img_files.sort()

dataset = [(os.path.join(img_path, f)) for f in img_files]

df_images = pd.DataFrame(dataset, columns=['images'])
df_images.to_csv('/home/antonio/workspace/ucl-thesis/csv/Patient017-Operator017-Novice-20230519-145030.csv', index=False)

output_folder = "/home/antonio/workspace/ucl-thesis/classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler/classification_with_progress_bar_Patient017-Operator017-Novice-20230519-145030"
os.makedirs(output_folder, exist_ok=True)

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
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.Tensor(image) / 255.0
        return image

df_images = pd.read_csv('/home/antonio/workspace/ucl-thesis/csv/Patient017-Operator017-Novice-20230519-145030.csv')
data_set = SegmentationDatasetVideoFrames(df_images)

# Load model
model = SegmentationModel().to(DEVICE)
model.load_state_dict(torch.load('classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler.pt'))

model.eval()  # Imposta il modello in modalità di valutazione

with torch.no_grad():
   
    # Inizializza una lista per contenere i colori delle barre di avanzamento
    progress_colors = []
    
    # Stampa le etichette predette sull'immagine e salva l'immagine modificata
    for idx in range(len(data_set)):
        
        image = data_set[idx]

        _, _, _, class_probs, _ = model(image.to(DEVICE).unsqueeze(0), None, None, None, None)  # (C, H, W) -> (1, C, H, W)
        
        # Applica una soglia alle probabilità per ottenere le etichette binarie
        binary_predictions = (class_probs > 0.5).int().item()  # Soglia 0.5
        
        # Aggiungi il colore della barra di avanzamento corrispondente alla predizione
        progress_colors.append('green' if binary_predictions else 'red')
        
        # Disegna le barre di avanzamento sopra l'immagine
        plt.imshow(image.numpy().transpose(1, 2, 0))

        # Disegna la barra di avanzamento in alto sopra l'immagine
        for i, color in enumerate(progress_colors):
            plt.axvline(x=i * image.shape[1] / len(data_set), color=color, linewidth= image.shape[1] / len(data_set), alpha=0.5, ymax=0.05)
        plt.axis('off')

        # Aggiorna la visualizzazione della figura
        plt.draw()

        # Salva l'immagine modificata nella cartella specifica
        output_path = os.path.join(output_folder, f"{idx}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
