import cv2
import torch
from math import log2

DATASET = ''
CSV = ''
CHECKPOINT_GEN = "generator.pth"
CHECKPOINT_CRITIC = "critic.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
CHANNELS_IMG = 3
Z_DIM = 256  # 512 in paper
IN_CHANNELS = 256  # 512 in paper
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
NUM_EPOCHS = 5
FIXED_NOISE = torch.randn(8, Z_DIM, 1, 1).to(DEVICE)
NUM_WORKERS = 4

# Hyperparameters etc
IMAGE_SIZE = 64
FEATURES_CRITIC = 64
FEATURES_GEN = 64
WEIGHT_CLIP = 0.01