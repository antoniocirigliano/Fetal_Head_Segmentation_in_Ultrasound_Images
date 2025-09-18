import cv2
import pandas as pd
import os

def add_black_border(image_path, mask_path, border_size):

    # Leggi l'immagine e la maschera
    image = cv2.imread(image_path)
    #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path)

    # Aggiungi il bordo nero all'immagine
    image_with_border = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Aggiungi il bordo nero alla maschera
    #mask_with_border = cv2.copyMakeBorder(mask, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0))
    mask_with_border = cv2.copyMakeBorder(mask, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image_with_border, mask_with_border

df = pd.read_csv('/home/antonio/workspace/ucl-thesis/csv/labeled_head.csv')

# Dimensione del bordo da aggiungere
border_size = 200

# Path per salvare le nuove immagini con i bordi neri in formato PNG
new_image_save_path = '/home/antonio/workspace/ucl-thesis/data/all_img_head_black_border'
# Path per salvare le nuove maschere in formato PNG
new_mask_save_path = '/home/antonio/workspace/ucl-thesis/data/all_masks_head_black_border'

# Creare una nuova lista di tuple (image_path, mask_path) con i bordi neri
augmented_data = []
for index, row in df.iterrows():
    image_path = row['images']
    mask_path = row['masks']

    augmented_image, augmented_mask = add_black_border(image_path, mask_path, border_size)

    # Ottenere solo il nome del file dalla path originale
    image_filename = os.path.basename(image_path)
    mask_filename = os.path.basename(mask_path)

    # Generare i percorsi per salvare le nuove immagini e maschere in formato PNG
    augmented_image_path = os.path.join(new_image_save_path, f'{os.path.splitext(image_filename)[0]}.png')
    augmented_mask_path = os.path.join(new_mask_save_path, f'{os.path.splitext(mask_filename)[0]}.png')

    # Salvare le nuove immagini e maschere con i bordi neri in formato PNG
    cv2.imwrite(augmented_image_path, augmented_image)
    cv2.imwrite(augmented_mask_path, augmented_mask)

    augmented_data.append((augmented_image_path, augmented_mask_path))
