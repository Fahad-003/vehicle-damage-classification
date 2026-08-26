# 🚗 Vehicle Damage Classification Using EfficientNetB0 Transfer Learning and Fine-Tuning

<p align="center">
  <img src="screenshots/home_page.png" alt="Vehicle Damage Classification" width="850">
</p>

An end-to-end **Deep Learning computer vision project** that classifies vehicle images into **7 different damage categories** using **EfficientNetB0 Transfer Learning and Fine-Tuning**. The project includes data cleaning, duplicate removal, class balancing, image augmentation, model evaluation, and deployment using **Streamlit**.

---

# 🚀 Project Links

* **🌐 Live Demo:** https://vehicle-damage-classification.streamlit.app/
* **📊 Dataset:** [Car Damage Assessment Dataset](https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment)

---

# 📌 Project Overview

Vehicle damage classification is an important computer vision application with potential use cases in **insurance assessment, vehicle inspection, and automated damage analysis**.

This project develops a multiclass image classification model capable of identifying seven common vehicle damage categories from images.

The workflow covers the complete deep learning pipeline, from **data cleaning and preprocessing to model training, fine-tuning, evaluation, and deployment**.

---

# 🎯 Project Objectives

* Clean and prepare the vehicle damage dataset
* Remove duplicate and inconsistent image records
* Handle class imbalance using class weights
* Apply image augmentation to improve generalization
* Build a multiclass image classification model
* Use transfer learning with EfficientNetB0
* Fine-tune the pretrained model
* Evaluate model performance using classification metrics
* Deploy the trained model using Streamlit

---

# 📂 Dataset Information

**Dataset:** [Car Damage Assessment Dataset](https://www.kaggle.com/datasets/hamzamanssor/car-damage-assessment)

### Dataset Processing

| Description                  |     Count |
| ---------------------------- | --------: |
| Original Dataset             | **1,594** |
| Damage Classification Images | **1,045** |
| Duplicate Records            |   **172** |
| Final Unique Images          |   **959** |
| Final Classes                |     **7** |

The original dataset contained an `unknown` class. Since these images did not represent a consistent damage category, the class was removed before model development.

### Damage Classes

| Class            | Images |
| ---------------- | -----: |
| `door_dent`      |    192 |
| `door_scratch`   |    154 |
| `glass_shatter`  |    136 |
| `tail_lamp`      |    135 |
| `head_lamp`      |    132 |
| `bumper_dent`    |    128 |
| `bumper_scratch` |     82 |

---

# ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* EfficientNetB0
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Pillow
* Streamlit

---

# 🔄 Project Workflow

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Remove Unknown Class
   │
   ▼
Duplicate Detection & Removal
   │
   ▼
Label Encoding
   │
   ▼
Stratified Train / Validation / Test Split
   │
   ▼
Data Augmentation
   │
   ▼
Class Weight Calculation
   │
   ▼
EfficientNetB0 Transfer Learning
   │
   ▼
Fine-Tuning
   │
   ▼
Model Evaluation
   │
   ▼
Model Saving
   │
   ▼
Streamlit Deployment
```

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

* Removed the `unknown` damage category.
* Detected duplicate image records using image hashing.
* Retained **959 unique damage images**.
* Encoded the seven damage categories into numerical labels.
* Used a **70:15:15 stratified split** for training, validation, and testing.
* Applied class weights to address class imbalance.

### Data Augmentation

Training images were augmented using:

* Horizontal flipping
* Random rotation
* Random zoom
* Random contrast

---

# 🤖 Model Architecture

The project uses **EfficientNetB0 pretrained on ImageNet** as the feature extraction backbone.

```text
Input Image (224 × 224 × 3)
          │
          ▼
   Data Augmentation
          │
          ▼
     EfficientNetB0
     (ImageNet Weights)
          │
          ▼
Global Average Pooling
          │
          ▼
       Dropout
          │
          ▼
   Dense + Softmax
          │
          ▼
     7 Damage Classes
```

### Training Strategy

The model was trained in two stages:

**1. Transfer Learning**

* EfficientNetB0 backbone initially frozen
* Learning rate: `0.001`

**2. Fine-Tuning**

* Top layers of EfficientNetB0 unfrozen
* Lower learning rate: `0.00001`
* Fine-tuned for the vehicle damage dataset

Balanced class weights were used during training to reduce the effect of class imbalance.

---

# 📈 Model Performance

The final model was evaluated on the held-out test set.

| Metric             |      Score |
| ------------------ | ---------: |
| **Test Accuracy**  | **75.00%** |
| **Macro F1-Score** | **0.7473** |

### Class-wise Performance

| Class            | Precision | Recall | F1-Score | Support |
| ---------------- | --------: | -----: | -------: | ------: |
| `bumper_dent`    |      0.85 |   0.58 |     0.69 |      19 |
| `bumper_scratch` |      0.48 |   0.77 |     0.59 |      13 |
| `door_dent`      |      0.77 |   0.69 |     0.73 |      29 |
| `door_scratch`   |      0.70 |   0.83 |     0.76 |      23 |
| `glass_shatter`  |      0.89 |   0.85 |     0.87 |      20 |
| `head_lamp`      |      0.71 |   0.75 |     0.73 |      20 |
| `tail_lamp`      |      0.94 |   0.80 |     0.86 |      20 |

> The model achieved its strongest performance on **glass shatter** and **head lamp** damage, while visually similar dent/scratch categories were comparatively more challenging.

---

# 📊 Visualizations

The notebook includes:

* Class distribution
* Training & validation accuracy
* Training & validation loss
* Confusion matrix
* Classification report
* Model evaluation metrics

---

# 💾 Model Deployment

The trained EfficientNetB0 model was integrated into a **Streamlit** application.

Users can:

* Upload a vehicle image
* Preview the image
* Generate a damage prediction
* View the predicted damage category

Run the application locally:

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```text
Vehicle-Damage-Assessment/
│
├── data/
│   ├── image/
│   └── data.csv
│
├── models/
│   ├── vehicle_damage_model.keras
│   ├── class_names.txt
│   └── label_encoder.joblib
│
├── notebook/
│   └── vehicle_damage_detection.ipynb
│
├── screenshots/
│   ├── home_page.png
│   └── prediction_result.png
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

# 🌟 Key Highlights

* End-to-End Deep Learning Project
* Multiclass Vehicle Damage Classification
* EfficientNetB0 Transfer Learning
* Fine-Tuning of Pretrained Model
* Duplicate Detection and Data Cleaning
* Class Imbalance Handling
* Image Data Augmentation
* **75.00% Test Accuracy**
* **0.7473 Macro F1-Score**
* Streamlit Deployment

---


# 👨‍💻 Author

**Fahad Rizvi**

* **GitHub:** https://github.com/Fahad-003
* **LinkedIn:** https://linkedin.com/in/fahadrizvi1

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.
