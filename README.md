MedXfer – Transfer Learning for Chest X-Ray Diagnosis

MedXfer is an educational machine learning project demonstrating the application of transfer learning to medical image classification using chest X-ray images. The project covers the full ML lifecycle, including model training, evaluation, backend inference via an API, and a frontend interface for visualization.

Disclaimer
This project is intended strictly for educational and research purposes. It is not a medical device and must not be used for clinical diagnosis or decision-making.

Project Overview

Medical imaging datasets are typically small, noisy, and costly to annotate. Training deep neural networks from scratch in this setting often leads to overfitting and poor generalization.

This project addresses these challenges using transfer learning:

A convolutional neural network pre-trained on ImageNet

Reuse of learned feature extraction layers

Retraining of the final classification layer on chest X-ray data

Binary classification: NORMAL vs PNEUMONIA

The project demonstrates:

Why transfer learning is effective for medical imaging

How to implement it in practice

How to deploy a trained model behind an API

How to visualize predictions in a frontend application

Transfer Learning Approach
Motivation

Medical datasets are limited in size

Training CNNs from scratch requires large-scale data

Early CNN layers learn general visual patterns such as edges, textures, and shapes

These features are transferable across domains

Implementation

Load an ImageNet-pretrained CNN (ResNet-18 or ResNet-50)

Freeze convolutional layers to act as a feature extractor

Replace the final fully connected layer

Train the new classifier on chest X-ray images

Optionally fine-tune deeper layers

This approach enables faster convergence, improved generalization, and reduced overfitting.

Project Structure
TransferLearning/
├── backend/                 FastAPI inference service
│   ├── app/
│   │   ├── main.py          API entry point
│   │   ├── inference.py    Model loading and prediction
│   │   └── schema.py       Response schemas
│   ├── model/
│   │   ├── model.pt        Trained model weights
│   │   └── model_meta.json Model metadata
│   └── requirements.txt
│
├── ml/                      Training and evaluation
│   ├── train.py             Transfer learning training script
│   ├── model_def.py         CNN architectures
│   ├── export_model.py      Model export for inference
│   ├── utils.py
│   ├── outputs/
│   │   ├── model.pt
│   │   └── model_meta.json
│   └── requirements.txt
│
├── frontend/                React frontend
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── api.js
│
├── chest_xray/              Dataset
│   ├── train/
│   ├── val/
│   └── test/
│
└── README.md

Dataset

Chest X-Ray Pneumonia Dataset

Classes: NORMAL, PNEUMONIA

Data split: Train, Validation, Test

Images are resized to 224×224 and normalized using ImageNet statistics to match the pretrained backbone

Model Details

Backbone: ResNet-18 (ImageNet pretrained)

Input size: 224 × 224

Output classes: 2

Loss function: Cross-Entropy

Optimizer: Adam

Evaluation metrics: Accuracy, AUROC, Precision, Recall, F1-Score

Evaluation Results (Test Set)
Metric	Value
Accuracy	84.1%
AUROC	0.936
F1-Score	0.81

Confusion Matrix:

[[138,  96],
 [  3, 387]]


The model demonstrates strong sensitivity for pneumonia detection with a reasonable precision–recall trade-off.

Backend API (FastAPI)

The backend handles:

Model loading

Image preprocessing

Inference and prediction

Endpoints
Method	Endpoint	Description
GET	/health	Health check
GET	/model-info	Model metadata
POST	/predict	Image prediction

Predictions return:

Predicted class label

Confidence score

Full class probability distribution

Frontend

The frontend is a React-based application that:

Allows users to upload chest X-ray images

Sends images to the backend API

Displays prediction results

Explains the transfer learning concept in a clear, educational format

Design principles:

Large, uncluttered layout

Medical-style UI

Left-aligned explanation section

Right-aligned prediction results

Running the Project Locally
1. Train the Model
cd ml
pip install -r requirements.txt
python train.py --data_dir ../chest_xray

2. Export the Model
python export_model.py

3. Run the Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

4. Run the Frontend
cd frontend
npm install
npm run dev

Learning Outcomes

This project demonstrates:

Practical transfer learning for medical imaging

Medical image preprocessing pipelines

Model evaluation using clinically relevant metrics

End-to-end AI system integration (ML → API → UI)

Author

Hadi Mostafa
Computer Engineering Student

Focus areas:

Artificial Intelligence

Computer Vision

Medical AI Systems
