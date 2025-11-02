import torch
from tqdm import tqdm

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, q_enc, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            q_enc = {k: v.squeeze(1).to(device) for k, v in q_enc.items()}
            outputs = model(imgs, q_enc)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total