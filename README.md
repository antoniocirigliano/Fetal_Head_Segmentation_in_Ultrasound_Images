# Fetal Head Segmentation in Ultrasound Images

This repository contains the code and experiments developed for my Master's thesis project, focused on **fetal head segmentation in prenatal ultrasound images** using deep learning.

The project explores different strategies based on the **UNet architecture**, including supervised, semi-supervised and multi-task learning, with extensions for both image and video processing.

---

## 📂 Repository Structure

```bash
fetal-head-segmentation-thesis/
├── Image_processing/        # Utilities and scripts for preprocessing and handling images
├── supervised_unet/         # Supervised UNet model training and evaluation
├── semi_supervised_unet/    # Semi-supervised UNet with labeled + unlabeled data
├── multitask_unet/          # Multi-task UNet for segmentation + classification
├── Video_processing/        # Scripts and methods for processing ultrasound videos
│
├── notebooks/               # Jupyter Notebooks for experiments and visualization
├── results/                 # Segmentation outputs, plots, and evaluation metrics
├── thesis/                  # LaTeX source of the thesis
│
├── README.md                # Project description
├── LICENSE                  # MIT License
└── requirements.txt         # Python dependencies
```

---

## 🧪 Models Overview

- **Supervised UNet**  
  Trained on a limited set of annotated images.  
  Loss: Dice Loss + Binary Cross Entropy.  

- **Semi-supervised UNet**  
  Combines annotated and unannotated images to improve generalization.  
  Loss: Labeled Loss + α * Unlabeled Loss.  

- **Multi-task UNet**  
  Performs segmentation and classification (head vs abdomen vs femur).  
  Loss: Labeled Loss + α * Unlabeled Loss + β * Classification Loss.  

---

## 📊 Results

- Semi-supervised and multi-task approaches improved robustness and generalization compared to purely supervised training.  
- Performance was evaluated using both quantitative metrics (Dice coefficient, IoU) and qualitative inspection of predicted masks.  
- Extensions to video processing allow analysis on ultrasound video sequences.  

---

## ⚙️ Requirements

The project is implemented in Python with:
- `pytorch`
- `pytorch-lightning`
- `segmentation-models-pytorch`
- `torchmetrics`
- `timm`
- `scikit-learn`
- `matplotlib`
- `numpy`
- `pandas`

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📜 License

This project is released under the MIT License.

