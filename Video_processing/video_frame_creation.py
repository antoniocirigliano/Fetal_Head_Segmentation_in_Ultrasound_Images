import os
import imageio
from moviepy.editor import VideoFileClip
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.getcwd()

# Percorso dei video in formato mp4
video_folder = "/home/antonio/workspace/ucl-thesis/video_chiara/"

# Percorso di output per i frames
frames_folder = "/home/antonio/workspace/ucl-thesis/video_frames_chiara/"

# Assicurati che la cartella di output esista
os.makedirs(frames_folder, exist_ok=True)

# Itera sui file mp4 nella cartella dei video
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        # Costruisci il percorso completo del video e dell'output
        video_path = os.path.join(video_folder, video_file)
        frames_out_dir = os.path.join(frames_folder, os.path.splitext(video_file)[0])

        # Assicurati che la cartella di output per il video corrente esista
        os.makedirs(frames_out_dir, exist_ok=True)

        # Apri il video utilizzando moviepy
        video_clip = VideoFileClip(video_path)

        # Estrai e salva ogni frame
        for idx, frame in enumerate(video_clip.iter_frames(fps=10, dtype="uint8")):
            frame_filename = os.path.join(frames_out_dir, f"{idx:04d}.png")
            imageio.imwrite(frame_filename, frame)

        # Chiudi il video clip
        video_clip.close()
