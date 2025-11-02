import torch

class Config:
    DATA_JSON = "data/osfstorage/VQA_RAD Dataset Public.json"
    IMG_DIR = "data/osfstorage/VQA_RAD Image Folder"
    MODEL_SAVE_PATH = "saved_models/medvqa_resnet_bert.pth"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 224
    MAX_LEN = 32
    BATCH_SIZE = 8
    EPOCHS = 3
    LR = 2e-5