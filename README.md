

# 🫁 MedXfer — Transfer Learning for Chest X-Ray Diagnosis

> **Educational AI project demonstrating how transfer learning can be applied to medical image classification using chest X-ray images.**
> This project covers the **full machine learning pipeline**: model training, evaluation, backend inference API, and a frontend visualization interface.

⚠️ **Disclaimer**: This project is for educational and research purposes only. It is **not a medical device** and must not be used for clinical decision-making.

---
<img width="885" height="893" alt="Screenshot 2025-12-22 152927" src="https://github.com/user-attachments/assets/11b62cd1-88f2-4315-b691-400593fbb892" />


## 📌 Project Overview

Medical imaging datasets are often **small, noisy, and expensive to label**. Training deep neural networks from scratch in such settings is inefficient and prone to overfitting.

To address this, **transfer learning** is used:

* A deep CNN pre-trained on **ImageNet**
* Feature extraction layers reused
* Final classification layer retrained on **Chest X-ray (NORMAL vs PNEUMONIA)**

This project demonstrates:

* Why transfer learning works
* How it is implemented in practice
* How trained models can be served via an API
* How predictions can be visualized in a frontend UI

---

## 🧠 Transfer Learning — Conceptual Explanation

### Why Transfer Learning?

* Medical datasets are usually **small**
* Training CNNs from scratch requires **millions of images**
* Early CNN layers learn **generic visual features**:

  * edges
  * textures
  * shapes
* These features are **universal**, not domain-specific

### How It Is Applied Here

1. Load a CNN pre-trained on **ImageNet** (ResNet-18 / ResNet-50)
2. Freeze convolutional layers (feature extractor)
3. Replace the final fully connected layer
4. Train only the new classifier on chest X-ray images
5. Optionally fine-tune deeper layers

This allows the model to:

* Learn faster
* Generalize better
* Avoid overfitting

---

## 🗂️ Project Structure

```
TransferLearning/
│
├── backend/                 # FastAPI inference service
│   ├── app/
│   │   ├── main.py          # API entry point
│   │   ├── inference.py    # Model loading & prediction
│   │   └── schema.py       # Response schemas
│   ├── model/
│   │   ├── model.pt        # Trained weights
│   │   └── model_meta.json # Model metadata
│   └── requirements.txt
│
├── ml/                      # Training & evaluation
│   ├── train.py             # Transfer learning training script
│   ├── model_def.py         # CNN architectures
│   ├── export_model.py      # Export trained model for inference
│   ├── utils.py
│   ├── outputs/
│   │   ├── model.pt
│   │   └── model_meta.json
│   └── requirements.txt
│
├── frontend/                # React UI
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── api.js
│
├── chest_xray/              # Dataset
│   ├── train/
│   ├── val/
│   └── test/
│
└── README.md
```

---

## 📊 Dataset

**Chest X-Ray Pneumonia Dataset**

Classes:

* `NORMAL`
* `PNEUMONIA`

Split:

* Train
* Validation
* Test

Images are resized and normalized using ImageNet statistics to match the pretrained backbone.

---

## 🧪 Model Architecture

* **Backbone**: ResNet-18 (ImageNet pretrained)
* **Input Size**: 224 × 224
* **Output**: 2 classes (Softmax)
* **Loss**: Cross-Entropy
* **Optimizer**: Adam
* **Metric**: AUROC, Precision, Recall, F1-Score

---

## 📈 Evaluation Results (Test Set)

| Metric           | Value     |
| ---------------- | --------- |
| Accuracy         | **84.1%** |
| AUROC            | **0.936** |
| F1-Score (Macro) | **0.81**  |

Confusion Matrix:

```
[[138,  96],
 [  3, 387]]
```

The model shows:

* High sensitivity for pneumonia detection
* Acceptable trade-off between precision and recall

---

## 🧠 Backend (FastAPI)

The backend provides:

* Model loading
* Image preprocessing
* Prediction endpoint

### Available Endpoints

| Method | Endpoint      | Description         |
| ------ | ------------- | ------------------- |
| GET    | `/health`     | API health check    |
| GET    | `/model-info` | Model metadata      |
| POST   | `/predict`    | Predict X-ray image |

Predictions return:

* Predicted label
* Confidence score
* Full class probability distribution

---

## 🎨 Frontend

The frontend is a **React-based interface** that:

* Allows image upload
* Sends images to backend API
* Displays prediction results visually
* Explains transfer learning concept in text form

UI focus:

* Large layout (no small cards)
* Clear medical-style design
* Left-aligned explanation panel
* Right-aligned prediction panel

---

## ▶️ Running Locally

### 1️⃣ Train Model

```bash
cd ml
pip install -r requirements.txt
python train.py --data_dir ../chest_xray
```

### 2️⃣ Export Model

```bash
python export_model.py
```

### 3️⃣ Run Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### 4️⃣ Run Frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 🎯 Learning Outcomes

This project demonstrates:

* Practical transfer learning implementation
* Medical image preprocessing
* Model evaluation using clinical metrics
* AI system integration (ML → API → UI)
* End-to-end ML system design

---

## 👤 Author

**Hadi Mostafa**
Computer Engineering Student
Focus areas:

* Artificial Intelligence
* Computer Vision
* Medical AI Systems

---

## 📜 License

This project is released for **educational and academic use only**.

---



