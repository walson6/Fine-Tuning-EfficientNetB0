# Automated Food Classification Using CNNs

**Jimmy Vu**  
jimmy.vu@sjsu.edu  
**Eric Zhao**  
eric.zhao@sjsu.edu  

---

## 1 Abstract
This project explores the application of Convolutional Neural Networks (CNNs) for automated food image classification using the Food-101 dataset [2]. Our objective is to improve upon the 56.4% baseline accuracy set by earlier models like AlexNet by fine-tuning modern pre-trained CNN architectures such as EfficientNetB0 [6]. We leverage transfer learning, data augmentation, and regularization techniques to address challenges like overfitting and dataset variability. Evaluation is based on standard metrics including accuracy, loss, and F1-score, with a target test accuracy above 80%. This project demonstrates the effectiveness of fine-tuned CNNs for real-world applications such as food tracking, dietary assessment, and restaurant recommendation systems.

---

## 2 Introduction
Previous work using traditional machine learning, such as Random Forests and early deep learning models like AlexNet, achieved moderate accuracy (56.4%) on the Food-101 dataset [2]. We aim to improve on this by using more advanced models to increase performance. Our approach leverages transfer learning, fine-tuning pre-trained CNN models on the Food-101 dataset, and applies techniques like data augmentation and dropout to prevent overfitting.

This work was completed as part of the final project for **CS171: Intro Machine Learning** at San Jose State University. This report documents the project and methods used.

---

## 3 Related Work

### 3.1 Traditional Machine Learning Approaches
Bossard et al. [2] applied a Random Forest algorithm combined with feature extraction techniques to classify food images. They used AlexNet as a baseline, which achieved 56.4% accuracy on the Food-101 dataset. While this result was a starting point, it highlighted the need for more advanced models.

### 3.2 Deep Learning, Transfer Learning, and EfficientNet
Liu et al. [4] explored the use of transfer learning for food image classification. They fine-tuned pre-trained models like AlexNet and GoogLeNet on food datasets. Their method improved classification accuracy while minimizing computational cost. We adopt a similar approach by fine-tuning **EfficientNetB0 [6]**, which balances computational cost and classification accuracy.

### 3.3 Data Augmentation
Liu et al. [4] and Huilgol [3] showed that data augmentation techniques such as random rotations, zoom, cropping, and brightness adjustment improve model generalization. This helps models handle variability in lighting, background, and camera angles.

---

## 4 Dataset and Features
The dataset used is the **Food-101 dataset [2]**:
- **Training set:** 68,175 images (67.5%)  
- **Validation set:** 7,575 images (7.5%)  
- **Test set:** 25,250 images (25%)  

Preprocessing steps included resizing, normalization, and data augmentation using techniques such as random rotation, shift, zoom, and flipping.

Dataset: [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)

---

## 5 Methods

### 5.1 Convolutional Neural Networks
CNNs apply convolutional filters to images to detect spatial hierarchies [6].

### 5.2 Transfer Learning
We initialized models with ImageNet-trained weights and fine-tuned them on the Food-101 dataset [4], [6].

### 5.3 EfficientNetB0
EfficientNetB0 [6] optimizes CNN scaling by uniformly adjusting depth, width, and resolution.

### 5.4 Data Augmentation and Preprocessing
We applied augmentation techniques (flipping, rotation, zooming, brightness adjustments) to improve model robustness [3], [4].

### 5.5 Loss Function and Optimization
- **Loss:** Categorical cross-entropy  
- **Optimizer:** Adam [7], learning rate = 0.0001

### 5.6 Evaluation Metrics
We evaluated performance using accuracy, loss, and F1-score across training, validation, and test sets.

---

## 6 Experiments / Results / Discussion

### 6.1 Quantitative Results
| Model              | Training Phase     | Test Accuracy | Test Loss |
|--------------------|-------------------|---------------|-----------|
| AlexNet (Baseline) | Baseline          | 56.4%         | -         |
| EfficientNetB0     | Initial (Frozen)  | 68.95%        | 1.1406    |
| EfficientNetB0     | Fine-tune block7a | 74.76%        | 0.9163    |
| EfficientNetB0     | Fine-tune block6d | 78.35%        | 0.7813    |
| EfficientNetB0     | Fine-tune block6a | 79.81%        | 0.7602    |
| EfficientNetB0     | Re-fine block7a   | 80.55%        | 0.7345    |
| EfficientNetB0     | Re-fine block6d   | **80.73%**    | 0.7364    |

### 6.2 Qualitative Results
We used **Grad-CAM** to visualize which image regions were most important for predictions. Results showed the model focused on discriminative regions of food items, confirming meaningful feature extraction.

### 6.3 Experiments
We also tested **ResNet50 [3]**, achieving **79.89%** accuracy after fine-tuning the last 40 layers. While slightly below EfficientNetB0, it proved to be a strong alternative.

---

## 7 Conclusion / Future Work
We achieved a peak accuracy of **80.73%** with EfficientNetB0, surpassing the AlexNet baseline of 56.4%. Grad-CAM visualizations confirmed the model’s interpretability. ResNet50 also performed strongly at 79.89%.

EfficientNetB0’s compound scaling enabled better performance with efficiency. Transfer learning and data augmentation were key to reducing overfitting.

**Future work:** explore ensemble methods, larger EfficientNet variants, and additional computational resources to push accuracy higher.

---

## 8 Contributions

### 8.1 Jimmy Vu
- Fixed and optimized data pipeline  
- Report writing and formatting  
- Trained EfficientNetB0 by fine-tuning different layers

### 8.2 Eric Zhao
- Wrote and reviewed reports  
- Led model training (EfficientNetB0, ResNet50)  
- Coordinated project workflow

---

## 9 References
1. MS Codes, "Keras: Use All CPU Cores"  
2. Bossard et al., *Food-101 – Mining Discriminative Components with Random Forests*, ECCV, 2014  
3. P. Huilgol, "Top 4 Pre-Trained Models for Image Classification with Python Code," Analytics Vidhya, 2020  
4. C. Liu et al., *Recognition of food images based on transfer learning and ensemble learning*, ResearchGate  
5. *Papers with Code - Food-101 Dataset*  
6. M. Tan and Q. V. Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML, 2019  
7. TensorFlow Developers, *TensorFlow Documentation*  
8. J. D. Hunter, *Matplotlib: A 2D graphics environment*, 2007  
9. Pedregosa et al., *Scikit-learn: Machine learning in Python*, JMLR, 2011  
10. Harris et al., *Array programming with NumPy*, Nature, 2020  
11. G. Bradski, *The OpenCV Library*, 2000  
12. The Pandas Development Team, *pandas-dev/pandas: Pandas (Version 1.0.5)*, Zenodo, 2020
