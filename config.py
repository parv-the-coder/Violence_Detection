"""
Centralized configuration for the Violence Detection project.
All hyperparameters, paths, and model settings in one place.
"""

import os

# ============================================================
# Paths
# ============================================================
# Absolute path to the project root directory
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

# Example structure for RAW_VIDEO_DIR (unzipped UCF-Crime dataset folders):
# Data/
#   Anomaly-Videos-Part-1/
#     Anomaly-Videos-Part-1/
#       Arrest/
#         Arrest001_x264.mp4
#   Training-Normal-Videos/
#     Normal_Videos_001_x264.mp4
RAW_VIDEO_DIR = os.path.join(PROJ_DIR, "Data")

# Directory where downsampled and deduplicated JPEG frames will be saved
# Example: ucf_dataset_proc5/Arrest/Arrest001_x264/0.jpg, 1.jpg, ...
FRAME_OUTPUT_DIR = os.path.join(PROJ_DIR, "ucf_dataset_proc5")

# Directory where the parsed training/testing dataset .npy chunks are saved
# Example: npy_data_output_15/train_img/0.npy, npy_data_output_15/train_lbl/0.npy, ...
NPY_DATA_DIR = os.path.join(PROJ_DIR, "npy_data_output_15")

# Directory where DINOv2 spatial feature embeddings are saved as numpy matrices
# Example: output_features_dinov2/train_features.npy, train_labels.npy
FEATURE_OUTPUT_DIR = os.path.join(PROJ_DIR, "output_features_dinov2")

# Path to the temporal annotations text file
ANNOTATION_FILE = os.path.join(PROJ_DIR, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")

# Directory where training history logs and the trained PyTorch models are stored
# Example: train_output/models/best_model.pt
TRAIN_OUTPUT_DIR = os.path.join(PROJ_DIR, "train_output")

# ============================================================
# DINOv2 Backbone Settings
# ============================================================
DINOV2_MODEL_NAME = "dinov2_vitb14"  # Base model: 86M params
FEATURE_DIM = 768  # DINOv2 vitb14 output dimension
IMAGE_SIZE = 224
NUM_CHANNELS = 3

# ImageNet normalization (used by DINOv2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# Temporal Model Settings
# ============================================================
SEQUENCE_LENGTH = 32  # Number of consecutive frames in sliding window
NUM_HEADS = 16  # Transformer attention heads
NUM_TRANSFORMER_LAYERS = 2  # Number of transformer encoder layers
DROPOUT = 0.3
NUM_CLASSES = 2  # Violence / Non-Violence

# ============================================================
# Training Settings
# ============================================================
BATCH_SIZE = 32
NUM_EPOCHS = 100
INITIAL_LR = 1e-3
LR_DECAY_RATE = 0.9
OPTIMIZER = "sgd"  # "sgd", "adam", or "rmsprop"
RANDOM_SEED = 42

# ============================================================
# Data Settings
# ============================================================
NPY_CHUNK_SIZE = 10000  # Frames per .npy chunk file
TEST_SPLIT_RATIO = 0.2
