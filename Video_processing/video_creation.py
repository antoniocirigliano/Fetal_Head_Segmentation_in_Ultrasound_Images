import cv2
import os

# Specifica la cartella contenente i frame
frames_folder = "/home/antonio/workspace/ucl-thesis/classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler/segmentation_and_classification_with_progress_bar_Operator_6"

# Ottieni la lista dei frame nella cartella
frame_files = sorted(os.listdir(frames_folder), key=lambda x: int(x.split(".")[0]))

# Apri il primo frame per ottenere le dimensioni del video
first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
height, width, _ = first_frame.shape

# Specifica il percorso di output per il video
output_video_path = "/home/antonio/workspace/ucl-thesis/classification-new-case-3_abdomen-femur-0.25_alpha-0.5_beta_1.0_epochs-150_lr-0.00003_batch-size-8_no-scheduler/segmentation_and_classification_Operator_6.mp4"

# Specifica il framerate del video
fps = 10  # Ricorda di specificare lo stesso framerate con cui hai estrapolato i frame

# Inizializza il writer video
video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Scrivi i frame nel video
for frame_file in frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    video_writer.write(frame)

# Rilascia il writer video
video_writer.release()
