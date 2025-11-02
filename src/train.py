import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.dataset import VQARadDataset
from src.model import MedVQA
from src.config import Config

def train():
    # Dataset Split
    dataset = VQARadDataset(Config.DATA_JSON, Config.IMG_DIR)
    records = dataset.records
    n = len(records)
    idx = list(range(n))
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

    train_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE)

    model = MedVQA(num_answers=len(dataset.answers)).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, q_enc, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
            imgs = imgs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            q_enc = {k: v.squeeze(1).to(Config.DEVICE) for k, v in q_enc.items()}

            optimizer.zero_grad()
            outputs = model(imgs, q_enc)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total:.3f}")

    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    train()