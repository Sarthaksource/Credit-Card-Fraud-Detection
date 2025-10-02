# Credit Card Fraud Detection System

A web application using machine learning to detect fraudulent credit card transactions in real-time.

## 🚀 Live Demo

🌐 **Live Application**: [https://credit-card-fraud-detection-m4dr.onrender.com](https://credit-card-fraud-detection-m4dr.onrender.com)

## 🌟 Key Features

-   **Multi-Model Approach**: Utilizes Logistic Regression, Random Forest, and XGBoost, combined into a powerful hybrid ensemble model for high accuracy.
-   **Interactive Web Interface**: Predict single transactions via a manual form or upload a CSV file for batch processing.
-   **Real-Time Results**: Get instant predictions with confidence scores and a clear interpretation of the result.
-   **Class Imbalance Handling**: Employs the SMOTE technique to create a balanced dataset for more effective model training.

## 📊 Performance Highlight

The final hybrid model achieves an accuracy of **99.97%** and an F1-Score of **85.11%** on the test set, demonstrating high effectiveness in identifying fraudulent transactions.

## 🔍 How to Use

### Manual Prediction
1.  Navigate to the **"Manual Entry"** tab.
2.  Fill in the transaction details (`Time`, `V1-V28` PCA features, and `Amount`).
3.  Click **"Predict Transaction"** to see the result.

### CSV Batch Prediction
1.  Navigate to the **"CSV Upload"** tab.
2.  Upload a CSV file containing the required columns: `Time, V1, V2, ..., V28, Amount`.
3.  Click **"Analyze CSV"** to view a summary and a detailed breakdown of predictions for each transaction.

## 🧠 How It Works

The machine learning pipeline involves three main stages:

1.  **Preprocessing**: Transaction data is normalized using Scikit-learn's `StandardScaler`.
2.  **Balancing**: The highly imbalanced dataset is balanced using the **SMOTE** (Synthetic Minority Oversampling Technique) to prevent model bias towards normal transactions.
3.  **Modeling**: Several models are trained on the preprocessed data. A final **Hybrid Model** (a soft-voting ensemble of Random Forest and XGBoost) is used for making the final predictions, as it provides the best overall performance.