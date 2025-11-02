import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

class VQARadDataset(Dataset):
    def __init__(self, json_path, img_dir):
        # --- Load JSON ---
        with open(json_path, 'r') as f:
            data = json.load(f)

        # --- Extract question–answer pairs ---
        self.records = []
        for entry in data:
            if "image_name" in entry and "question" in entry and "answer" in entry:
                self.records.append({
                    "image": entry["image_name"],
                    "question": entry["question"],
                    "answer": entry["answer"]
                })

        self.img_dir = img_dir

        # --- Normalize all answers to strings and clean ---
        answers_clean = []
        for r in self.records:
            ans = str(r["answer"]).strip().lower() if r["answer"] is not None else "unknown"
            # Handle NULL or empty answers
            if ans in ["", "null", "none", "nan"]:
                ans = "unknown"
            answers_clean.append(ans)

        self.answers = sorted(set(answers_clean))
        self.ans2idx = {a: i for i, a in enumerate(self.answers)}
        self.idx2ans = {i: a for a, i in self.ans2idx.items()}

        # --- Tokenizer & Transform ---
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = os.path.join(self.img_dir, record["image"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(image)

        # Tokenize question
        question_enc = self.tokenizer(
            record["question"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=32
        )

        # Normalize and map answer to index
        ans = str(record["answer"]).strip().lower()
        if ans in ["", "null", "none", "nan"]:
            ans = "unknown"
        label = self.ans2idx.get(ans, 0)

        return img_tensor, question_enc, label


# 🔍 Test the dataset independently
if __name__ == "__main__":
    dataset = VQARadDataset(
        json_path="data/osfstorage/VQA_RAD Dataset Public.json",
        img_dir="data/osfstorage/VQA_RAD Image Folder"
    )

    print("✅ Total Samples:", len(dataset))
    img, q_enc, label = dataset[0]
    print("🩻 Image shape:", img.shape)
    print("❓ Question:", dataset.records[0]["question"])
    print("💬 Answer:", dataset.records[0]["answer"])