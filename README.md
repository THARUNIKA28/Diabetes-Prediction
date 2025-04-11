# 🩺 Diabetes Prediction using Machine Learning

This project uses machine learning techniques to predict the likelihood of diabetes in patients based on medical data. It demonstrates the full data science pipeline — from data exploration and preprocessing to model training, evaluation, and visualization.

## 📊 Overview

The goal is to build a binary classification model that can predict whether a person is diabetic or not using features like glucose levels, BMI, pregnancies, and more. This project uses the **Pima Indians Diabetes Dataset**, a widely used dataset for health diagnostics.

## 🧠 Models Used

- **Logistic Regression** – interpretable baseline model for binary classification.
- **Decision Tree Classifier** – non-linear model for higher flexibility and insight into decision paths.

## 🛠️ Technologies & Tools

- **Python**
- **Pandas**, **NumPy** – for data manipulation
- **Matplotlib**, **Seaborn** – for data visualization
- **Scikit-learn** – for machine learning modeling and evaluation
- **Google Colab** – for cloud-based notebook execution

## 🔬 Key Steps

1. **Data Loading** from Google Drive.
2. **Exploratory Data Analysis (EDA)**:
   - Descriptive stats, correlation heatmaps, pairplots.
   - Box plots to understand distribution and outliers.
3. **Data Cleaning**:
   - Outlier capping at the 99th percentile.
4. **Feature Engineering**:
   - Feature-target split and train-test partitioning.
5. **Model Training**:
   - Logistic Regression and Decision Tree Classifier.
6. **Evaluation**:
   - Accuracy Score, Classification Report, Confusion Matrix.
7. **Visualization**:
   - Accuracy comparison, correlation matrix, and model insights.

## 📈 Results

| Model                | Accuracy |
|---------------------|----------|
| Logistic Regression | 75%  |
| Decision Tree       | 74%  |



## 🔗 Dataset

- [Pima Indians Diabetes Dataset (Kaggle)](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## 🚀 Future Enhancements

- Add **Random Forest** and **XGBoost** for improved accuracy.
- Deploy the model using **Streamlit** or **Flask** for real-time prediction.
- Hyperparameter tuning and cross-validation.


## 🙌 Acknowledgements

- UCI & Kaggle for the dataset
- Scikit-learn community for accessible machine learning tools

---

**✨ This project showcases my skills in data preprocessing, model building, and visualization — essential for AI-driven healthcare solutions **
