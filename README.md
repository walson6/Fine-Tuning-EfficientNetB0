# Fine-Tuning EfficientNetB0 for Food-101 Classification

This repository contains the final project for **CS171: Intro to Machine Learning** at San Jose State University.  
The project explores **CNN-based food image classification** using the **Food-101 dataset**.

---

## 📌 Overview
- **Goal:** Improve AlexNet baseline (56.4%) for food classification
- **Approach:** Transfer learning with EfficientNetB0 and ResNet50
- **Techniques:** Data augmentation, fine-tuning, regularization
- **Result:** Achieved **80.73% test accuracy** with EfficientNetB0

---

## ⚙️ Setup & Usage
### 1. Clone the repo
```bash
git clone https://github.com/walson6/Fine-Tuning-EfficientNetB0.git
cd Fine-Tuning-EfficientNetB0
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook your_notebook.ipynb
```
Follow the notebook cells to train and evaluate the model.

---

## 📖 Full Report
Read the complete project write-up here: [`food_101_cnn_report.md`](./food_101_cnn_report.md).

---

## 👥 Contributors
- **Jimmy Vu** – Data pipeline, EfficientNetB0 experiments, report writing
- **Eric Zhao** – Model training, ResNet50 experiments, project coordination

---

## 📚 References
- Bossard et al., *Food-101 – Mining Discriminative Components with Random Forests*, ECCV 2014
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019
- TensorFlow Documentation
- Additional references in the [full report](./food101_cnn_report.md)
