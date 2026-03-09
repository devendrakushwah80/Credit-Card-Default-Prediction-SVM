# 💳 Credit Card Default Prediction using SVM

This project builds a **machine learning model using Support Vector Machine (SVM)** to predict whether a credit card customer is likely to default on their payment next month.

The focus is on **clean preprocessing, proper feature scaling, class imbalance handling, and robust evaluation**, following industry best practices.

---

## 📊 Dataset Overview

The dataset contains customer demographic details, credit history, bill amounts, and past payment behavior.

**Target Variable**
- `default payment next month`
  - `1` → Default
  - `0` → No Default

---

## 🧠 Project Workflow

### 1️⃣ Data Cleaning
- Fixed header misalignment
- Removed unnecessary `ID` column
- Converted all features to numeric format
- Checked and removed duplicate records

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Statistical summary using `describe()`
- Missing value checks
- Boxplots for outlier detection
- Histograms for feature distribution analysis

---

### 3️⃣ Outlier Handling
- Used **IQR (Interquartile Range) method**
- Calculated lower and upper bounds
- Filtered extreme values for numerical stability

---

### 4️⃣ Feature Engineering & Preprocessing
- Separated features and target
- Applied **StandardScaler** using `ColumnTransformer`
- Ensured no data leakage by fitting scaler only on training data

---

### 5️⃣ Model Building
- Algorithm: **Support Vector Machine (SVM)**
- Kernel: `RBF`
- Parameters:
  - `C = 1`
  - `gamma = 'scale'`
  - `class_weight = 'balanced'` (to handle class imbalance)

---

### 6️⃣ Model Evaluation
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- ROC Curve
- ROC–AUC Score

The model demonstrates strong discrimination capability while maintaining balance between precision and recall.

---

## 📈 Results

- **ROC–AUC Score:** ~0.77  
- Improved minority class recall using class weighting
- Stable and realistic performance (no data leakage)

---

## 🛠 Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib & Seaborn
- Scikit-learn

---

## 📁 Project Structure

├── credit_card_default_prediction_svm.ipynb
├── credit_default.csv
├── README.md
├── requirements.txt


---

## 🚀 Key Learnings

- Importance of proper feature scaling for SVM
- Handling class imbalance using `class_weight`
- Why ROC–AUC is critical for imbalanced classification problems
- End-to-end ML pipeline design

---

## 📌 Future Improvements
- Hyperparameter tuning using GridSearchCV (Done)
- Try other kernels (linear, polynomial)
- Compare with Logistic Regression and XGBoost
- Feature selection techniques (RFE, ANOVA)

---

## 🤝 Author
**Devendra Kushwah**

If you find this project useful, feel free to ⭐ the repository!
