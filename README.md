# Medical Visual Question Answering (VQA-RAD)

This project implements an end-to-end **Medical Visual Question Answering (VQA)** system using the **VQA-RAD** dataset. The model takes a medical image (e.g., X-ray, MRI, CT scan) and a natural language question as input, then generates an appropriate answer. It combines computer vision and natural language understanding techniques to enable multimodal reasoning within the medical domain.

---

## 1. Project Overview

The goal of this project is to build a deep learning system that:

* Understands medical questions in natural language.
* Analyzes medical images (radiology scans).
* Predicts accurate answers based on the visual and textual context.

This is achieved through a **ResNet-50 + BERT fusion model**, which jointly encodes the image and question features and classifies the answer from a learned vocabulary.

---

## 2. Dataset

**Dataset Used:** [VQA-RAD (Radiology Visual Question Answering Dataset)](https://osf.io/89kps/files)

**Description:**

* Contains 315 medical images and over 3,500 question-answer pairs.
* Covers a range of modalities such as CT, MRI, and X-ray.
* Includes both **closed-ended** (Yes/No) and **open-ended** questions.

**Example:**

| Image           | Question                            | Answer |
| --------------- | ----------------------------------- | ------ |
| synpic54610.jpg | Are regions of the brain infarcted? | Yes    |
| synpic29265.jpg | Is the lung expanded?               | No     |

---

## 3. System Architecture

### **Model Pipeline**

```
Image → ResNet50 → Visual Embedding
Question → BERT → Text Embedding
Concatenation → Fully Connected Layers → Answer Prediction
```

### **Core Components**

| Component          | Description                                            |
| ------------------ | ------------------------------------------------------ |
| **Visual Encoder** | Pretrained ResNet-50 extracts 2048-dim image features. |
| **Text Encoder**   | BERT-base-uncased encodes question semantics.          |
| **Fusion Layer**   | Concatenates embeddings and passes through an MLP.     |
| **Classifier**     | Outputs softmax probabilities across possible answers. |

---

## 4. Folder Structure

```
VQA-RADIOLOGY/
│
├── data/
│   └── osfstorage/
│       ├── VQA_RAD Image Folder/
│       ├── VQA_RAD Dataset Public.json
│       └── Readme.docx
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
└── README.md
```

---

## 5. Setup Instructions

### **Step 1. Environment Setup**

Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate     # Windows
source .venv/bin/activate    # Linux/Mac
```

### **Step 2. Install Dependencies**

```bash
pip install -r app/requirements.txt
```

### **Step 3. Dataset Placement**

Ensure the dataset files are structured as follows:

```
data/osfstorage/
  ├── VQA_RAD Dataset Public.json
  └── VQA_RAD Image Folder/
```

### **Step 4. Train the Model**

Run the training pipeline from the project root:

```bash
python -m src.train
```

### **Step 5. Evaluate and Test**

After training completes, evaluate performance or run predictions:

```bash
python -m src.inference
```

---

## 6. Streamlit Application

An interactive interface for testing your model.

**Run the app:**

```bash
streamlit run app/app.py
```

**Features:**

* Upload a medical image (JPG or PNG).
* Type a question about the image.
* The app displays the predicted answer and image preview.

---

## 7. Explainability

Grad-CAM is used to visualize the model’s attention over the medical image, highlighting areas influencing the prediction.
See implementation in `src/explainability.py`:

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
```

---

## 8. Key Python Files

| File                | Description                                      |
| ------------------- | ------------------------------------------------ |
| `config.py`         | Stores configuration paths and hyperparameters.  |
| `dataset.py`        | Loads and preprocesses the VQA-RAD dataset.      |
| `model.py`          | Defines the ResNet50 + BERT fusion architecture. |
| `train.py`          | Trains the model and saves checkpoints.          |
| `inference.py`      | Performs question answering inference.           |
| `explainability.py` | Generates Grad-CAM heatmaps.                     |
| `app.py`            | Streamlit web interface.                         |

---

## 9. Requirements

```
torch
torchvision
transformers
datasets
pandas
numpy
pillow
tqdm
scikit-learn
matplotlib
streamlit
grad-cam
```

---

## 10. Future Enhancements

* Integrate **BioViL-T** or **BLIP-2-Med** for transformer-based multimodal reasoning.
* Extend to additional datasets such as **PathVQA** or **VQA-Med 2021**.
* Add **sequence-to-sequence answer generation** for open-ended questions.
* Implement **attention visualization** for better interpretability.

---

## 11. References

* Lau, J. J., Gayen, S., Ben Abacha, A., & Demner-Fushman, D. (2018).
  *A dataset of clinically generated visual questions and answers about radiology images (VQA-RAD)*.
  arXiv preprint arXiv:1807.10221.

* Hugging Face Dataset Card: [https://huggingface.co/datasets/flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)