# 🩻 Visual Question Answering in Radiology (VQA-RAD)

An **end-to-end Medical Visual Question Answering (VQA)** system built using the **VQA-RAD** dataset.
This project demonstrates how to combine **deep learning for images (ResNet)** and **language models (BERT)** to answer natural-language questions about medical images (e.g., “Are regions of the brain infarcted?”).

---

## 🧩 Project Overview

The goal of this project is to enable a model to read and interpret radiology images—such as X-rays, MRIs, and CT scans—and respond accurately to questions in plain English.

### 🔍 Key Objectives

* Preprocess and normalize the VQA-RAD dataset.
* Build a multimodal deep learning model combining visual and textual representations.
* Train and evaluate the model on the medical Q&A pairs.
* Deploy the trained model through an interactive **Streamlit web app**.

---

## 🏗️ Architecture

| Component           | Description                                                                               |
| ------------------- | ----------------------------------------------------------------------------------------- |
| **Visual Encoder**  | A pretrained **ResNet-50** extracts image features from medical scans.                    |
| **Text Encoder**    | A **BERT** model converts natural-language questions into vector embeddings.              |
| **Fusion Layer**    | Image and question embeddings are concatenated and passed through fully connected layers. |
| **Classifier Head** | Predicts the most likely answer from a fixed vocabulary of unique answers.                |

---

## 🗂️ Folder Structure

```
VQA-RADIOLOGY/
│
├── data/
│   └── osfstorage/
│       ├── VQA_RAD Image Folder/
│       ├── VQA_RAD Dataset Public.json
│       ├── VQA_RAD Dataset Public.xlsx
│       └── VQA_RAD Dataset Public.xml
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── explainability.py
│   └── utils.py
│
├── saved_models/
│   └── medvqa_resnet_bert.pth
│
├── app/
│   ├── app.py
│   └── requirements.txt
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Create Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r app/requirements.txt
```

### 3️⃣ Verify Dataset Paths

Ensure the paths in `src/config.py` match your folder layout:

```python
class Config:
    DATA_JSON = "data/osfstorage/VQA_RAD Dataset Public.json"
    IMG_DIR = "data/osfstorage/VQA_RAD Image Folder"
```

---

## 🧠 Model Training

Train the model from the project root:

```bash
python -m src.train
```

During training you’ll see:

```
Epoch 1/3 | Loss: 1.82 | Acc: 0.47
Epoch 2/3 | Loss: 1.15 | Acc: 0.61
...
✅ Model saved successfully.
```

Trained weights are stored under `saved_models/medvqa_resnet_bert.pth`.

---

## 📈 Evaluation

Evaluate validation accuracy via:

```bash
python -m src.evaluate
```

Metrics such as overall accuracy, per-answer precision, and class distribution can be extended in `evaluate.py`.

---

## 🔍 Inference

Run predictions directly:

```python
from src.inference import ask_question
answer = ask_question("data/osfstorage/VQA_RAD Image Folder/synpic54610.jpg",
                      "Are regions of the brain infarcted?")
print("Predicted Answer:", answer)
```

---

## 🔬 Explainability (Grad-CAM)

Use Grad-CAM to visualize which regions influence predictions:

```python
from src.explainability import visualize_cam
visualize_cam(model, input_tensor, [model.vision.layer4[-1]])
```

This generates heatmaps highlighting clinically relevant areas.

---

## 🌐 Web Application

Run the Streamlit interface:

```bash
streamlit run app/app.py
```

### UI Features

* Upload a medical image (`.jpg`/`.png`)
* Enter a natural-language question
* View the model’s predicted answer
* Visualize image and attention (optional)

---

## 🧩 Key Files Explained

| File                | Purpose                                                          |
| ------------------- | ---------------------------------------------------------------- |
| `dataset.py`        | Loads JSON annotations, tokenizes text, and preprocesses images. |
| `model.py`          | Defines the multimodal ResNet + BERT architecture.               |
| `train.py`          | Training loop with loss computation and accuracy logging.        |
| `evaluate.py`       | Validation and metric functions.                                 |
| `inference.py`      | Simple function for testing trained models.                      |
| `explainability.py` | Implements Grad-CAM visualizations.                              |
| `app/app.py`        | Streamlit-based demo interface.                                  |

---

## 🧮 Dataset Information

**Dataset:** [VQA-RAD (Open Science Framework)](https://osf.io/89kps/)

* **Images:** ~315 radiology scans (CT, MRI, X-Ray)
* **QA pairs:** ~3,500 question–answer pairs
* **Domains:** Chest, Abdomen, Head
* **Answer types:** Binary (yes/no) and descriptive (organ names, conditions)

---

## 🧱 Technologies Used

| Category            | Tools                             |
| ------------------- | --------------------------------- |
| **Language Model**  | BERT (`bert-base-uncased`)        |
| **Vision Model**    | ResNet-50 (`torchvision.models`)  |
| **Frameworks**      | PyTorch, HuggingFace Transformers |
| **Visualization**   | Matplotlib, Grad-CAM              |
| **Deployment**      | Streamlit                         |
| **Data Processing** | Pandas, NumPy                     |

---

## 🚀 Future Improvements

* 🔄 Replace BERT + ResNet with **BLIP-2** or **LLaVA-Med** for open-ended answers.
* 🩸 Add **medical ontology grounding** (UMLS, RadLex).
* 🌍 Merge **PathVQA**, **SLAKE**, and **VQA-Med** for larger multimodal training.
* 🧾 Integrate **explainable reports** with textual rationale generation.

---

## 🧑‍💻 Contributors

* **Project Lead:** *Saranga Kumarapeli*
* **Dataset Source:** *Lau et al., VQA-RAD Dataset*
* **Frameworks Used:** PyTorch | HuggingFace | Streamlit

---

## 📜 License

This project is released under the **MIT License**.
The dataset (VQA-RAD) follows the **CC0 1.0 Universal License** as provided by OSF.

---

## 💬 Acknowledgements

* *VQA-RAD: Visual Question Answering in Radiology* (Lau et al., 2018)
* Hugging Face for model backbones
* PyTorch and Streamlit communities

---

> 🧩 *This repository demonstrates how multimodal AI can aid clinical reasoning by bridging computer vision and natural language understanding in radiology.*