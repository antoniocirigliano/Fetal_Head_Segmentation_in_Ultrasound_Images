import os
from PIL import Image

def create_black_mask(input_folder, output_folder):
    
    # Controlla se la cartella di output esiste, altrimenti creala
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Scansiona tutte le immagini nella cartella di input
    for filename in os.listdir(input_folder):
        # Costruisci il percorso completo dell'immagine e della maschera
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Apri l'immagine e ottieni le dimensioni
        image = Image.open(input_path)
        width, height = image.size

        # Crea una nuova immagine completamente nera con le stesse dimensioni
        black_mask = Image.new('L', (width, height), color=0)

        # Salva la maschera nera nell'output folder
        black_mask.save(output_path)

input_folder = '/home/antonio/workspace/ucl-thesis/data/img_abdomen_femur'
output_folder = '/home/antonio/workspace/ucl-thesis/data/masks_abdomen_femur'

create_black_mask(input_folder, output_folder)