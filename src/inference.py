import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
from src.model import MedVQA
from src.config import Config
from src.dataset import VQARadDataset

dataset = VQARadDataset(Config.DATA_JSON, Config.IMG_DIR)
model = MedVQA(num_answers=len(dataset.answers))
model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE))
model.to(Config.DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def ask_question(image_path, question):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(Config.DEVICE)
    q_enc = tokenizer(question, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=Config.MAX_LEN)
    q_enc = {k: v.to(Config.DEVICE) for k, v in q_enc.items()}

    with torch.no_grad():
        logits = model(img, q_enc)
    pred = logits.argmax(1).item()
    return dataset.idx2ans[pred]