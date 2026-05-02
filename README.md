# Fetal Health Classification using MLP

This project implements a complete end-to-end machine learning pipeline for fetal health classification using a Multi-Layer Perceptron (MLP).

The objective of this project is to classify fetal health into three categories:
- Normal
- Suspect
- Pathological

The project follows best practices in data preprocessing, feature selection, model training, evaluation, and experiment tracking using Weights & Biases (W&B).

---

## 📌 Project Overview

This repository was developed as part of an academic project focused on building a robust and reproducible machine learning pipeline.

The pipeline ensures:
- Data quality and validation
- Reproducibility through configuration and seeds
- Proper feature selection and preprocessing
- Model training with early stopping
- Performance evaluation using multiple metrics
- Experiment tracking and versioning with W&B

---

## ⚙️ Complete Pipeline

The project follows these steps:

### 1. Data Ingestion
- Download dataset (Kaggle)
- Load raw CSV data

### 2. Data Cleaning
- Remove duplicate rows
- Check for missing values
- (Outliers considered but not removed due to data loss)

### 3. Data Validation
- Ensure no missing values
- Validate target values (1, 2, 3)
- Confirm no duplicates remain

### 4. Train/Test Split
- Stratified split (preserves class distribution)
- 80% training / 20% testing

### 5. Statistical Validation
- Kolmogorov–Smirnov (KS) test
- Ensures train/test distributions are similar

### 6. Feature Selection
Three methods combined:
- Spearman Correlation
- Mutual Information
- Random Forest Feature Importance

Then:
- Normalize scores
- Compute final ranking
- Select Top-K features (K=10)
- Remove multicollinearity using VIF

Final result:
- 7 selected features

### 7. Preprocessing
- Standardization using StandardScaler
  - Fit only on training data (avoids leakage)
- Convert to PyTorch tensors
- Create DataLoaders

### 8. Model (MLP)
Architecture:
- Input: 7 features
- Hidden layers: [64, 32]
- Activation: ReLU
- Dropout: 0.2
- Output: 3 classes

### 9. Training
- Loss: CrossEntropyLoss (multiclass)
- Optimizer: Adam
- Batch size: 32
- Epochs: 100
- Early stopping (patience=10)

### 10. Experiment Tracking (W&B)
- Logs:
  - train_loss
  - val_loss
  - accuracy
  - f1_score
- Stores:
  - hyperparameters
  - datasets (artifacts)
  - trained model (.pt)

### 11. Evaluation
Metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Final performance:
- Accuracy: ~91%
- F1-score: ~0.90

---

## 📊 Key Results

| Metric | Value |
|------|------|
| Accuracy | 0.91 |
| Weighted F1-score | 0.90 |

Observations:
- Strong performance on "Normal"
- Good detection of "Pathological"
- Lower recall for "Suspect" due to class imbalance and overlapping features

---

## 🧠 Model Explanation

The model uses:
- ReLU activation (avoids vanishing gradient)
- Dropout (prevents overfitting)
- CrossEntropyLoss (multiclass classification)

No Softmax is applied in the model, as it is internally handled by the loss function.

---

## 📁 Project Structure

mlp-advanced-ai-topics/
│
├── config/
│ └── config.yaml
│
├── data/
│ ├── raw/
│ └── processed/
│
├── pipelines/
│ └── run_pipeline.py
│
├── src/
│ ├── data_cleaning.py
│ ├── data_loading.py
│ ├── data_validation.py
│ ├── feature_selection.py
│ ├── model.py
│ ├── preprocessing.py
│ ├── split_data.py
│ ├── train.py
│ └── utils.py
│
├── reports/
│ └── figures/
│ └── confusion_matrix.png
│
├── tests/
│ ├── test_cleaning.py
│ └── test_split.py
│
├── best_model.pt
├── fetal_health_notebook.ipynb
├── README.md
└── requirements.txt

---

## ▶️ How to Run

### 1. Create virtual environment

```bash
python -m venv .venv
```
Activate:

```bash
#### Windows
.venv\Scripts\activate
```

```bash
#### Linux / Mac
source .venv/bin/activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Login to Weights & Biases

```bash
wandb login
```
Paste your API key.

---

### 4. Kaggle Authentication (for dataset download)

This project uses `kagglehub` to download the dataset from Kaggle.

Make sure you are authenticated before running the pipeline.

#### Option 1: Using Kaggle CLI

1. Create an account on https://www.kaggle.com
2. Go to your account settings and generate an API token
3. Place the `kaggle.json` file in:

- Windows:
``` bash
C:\Users\YOUR_USERNAME.kaggle\
```

- Linux / Mac:

``` bash
~/.kaggle/
```

### Option 2: Using environment variables

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

### 4. 🚀 Run the Project

You can execute the project in two different ways:

- Option 1: Run the full pipeline

```bash
python pipelines/run_pipeline.py
```
- Option 2: Run using VS Code (Jupyter)

1. Open the project in VS Code:

``` bash
cd mlp-advanced-ai-topics
```
```bash
code .
```
2. Open fetal_health_notebook.ipynb
3. Select the .venv kernel
4. Run the cells using Shift + Enter

---

## 🔁 Reproducibility

- Random seed fixed
- Config file controls all parameters
- No data leakage (scaler fit only on train)
- Full pipeline automated

---

## 📈 W&B Tracking

This project uses W&B for:

- Logging training metrics
- Tracking hyperparameters
- Versioning datasets and models

Artifacts created:
- raw_data
- clean_data
- dataset_split
- best_model (multiple versions)

---

## ⚠️ Known Limitations

- Class imbalance affects "Suspect" recall
- Feature overlap between classes
- MLP may not capture all nonlinear patterns

---

## 🚀 Future Improvements

- Apply class weighting
- Oversampling (SMOTE)
- Hyperparameter tuning
- Try other models (XGBoost, LightGBM)
- Deeper neural networks

---

## 📚 Technologies Used

- Python
- PyTorch
- Scikit-learn
- Pandas / NumPy
- Seaborn / Matplotlib
- Weights & Biases (W&B)

---

## 📌 Conclusion

This project demonstrates a complete and reproducible machine learning workflow, from raw data to model evaluation, following best practices in MLOps and experimental tracking.