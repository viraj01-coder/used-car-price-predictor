# 🚗 Used Car Price Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

A machine learning project that predicts used car prices using Ridge Regression
with Polynomial Features. The goal is to identify key factors affecting car prices
and build an accurate, deployable prediction model.

---

## 🌐 Live Demo

Check out the interactive Streamlit app:

**[Launch App](https://used-car-price-predictor-6g2wersaihrvzufyy7xjne.streamlit.app/)**

---

## 📊 Table of Contents

- [Project Overview](#-project-overview)
- [Live Demo](#-live-demo)
- [Dataset](#-dataset)
- [Tools & Libraries](#️-tools--libraries)
- [Project Structure](#-project-structure)
- [What I Did](#-what-i-did)
- [Key Findings](#-key-findings)
- [Model Performance](#-model-performance)
- [How to Run Locally](#️-how-to-run-locally)

---

## 📖 Project Overview

This project performs end-to-end analysis on a UK used car dataset.
It covers data cleaning, exploratory data analysis, feature correlation,
machine learning model training with hyperparameter tuning via GridSearchCV,
and a live Streamlit web app with real-time price prediction and multi-currency support (GBP, USD, EUR, INR).

---

## 📦 Dataset

- **Source:** IBM Skills Network (Used Car Price Analysis)
- **Records:** ~18,000 cars
- **Domain:** UK Used Car Market (Ford models)
- **Features:** Year, Mileage, Fuel Type, Transmission, Tax, MPG, Engine Size, Price

---

## 🛠️ Tools & Libraries

| Tool / Library | Purpose |
|----------------|---------|
| Python 3 | Core programming language |
| Pandas | Data manipulation and analysis |
| Matplotlib | Static visualizations |
| Seaborn | Statistical visualizations |
| Plotly Express | Interactive visualizations |
| scikit-learn | Machine learning model training |
| Streamlit | Interactive web app |
| Jupyter Notebook | Analysis and exploration |

---

## 📂 Project Structure

```
used-car-price-predictor/
│
├── car_price.ipynb                  # Main analysis and model training notebook
├── app.py                           # Streamlit web app
├── used_car_price_analysis.csv      # Dataset
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## ✅ What I Did

- 🔹 Performed Data Cleaning — handled missing values in tax column, removed duplicates
- 🔹 Identified top 5 features correlated with price using correlation analysis
- 🔹 Performed EDA using Matplotlib and Seaborn — price vs year, mileage, engine size
- 🔹 Trained Polynomial Ridge Regression model (degree=2) with StandardScaler pipeline
- 🔹 Used GridSearchCV with 4-fold cross-validation to find optimal alpha = 100
- 🔹 Deployed Streamlit app with real-time price prediction and multi-currency support (GBP, USD, EUR, INR)

---

## 🔍 Key Findings

1. **Year of manufacture** is the strongest price predictor — correlation of 0.636
2. **Mileage** is the second most important factor — higher mileage = lower price (0.531)
3. **Engine size** has a positive impact on price — larger engines cost more (0.411)
4. **Tax** also correlates with pricing patterns (0.406)
5. **Polynomial Ridge Regression** (degree=2) significantly outperforms simple linear models
6. **Optimal alpha = 100** found via GridSearchCV with 4-fold cross-validation

---

## 🤖 Model Performance

| Model | R² Score |
|-------|----------|
| Ridge Regression (α=0.1) | ~0.68 |
| Polynomial Ridge (degree=2, α=0.1) | 0.7557 |
| Polynomial Ridge (degree=2, α=100) ✅ | **0.7560** |

---

## ⚙️ How to Run Locally

1. Clone this repository
   ```bash
   git clone https://github.com/viraj01-coder/used-car-price-predictor.git
   cd used-car-price-predictor
   ```

2. Install required libraries
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

4. Open the notebook
   ```bash
   jupyter notebook car_price.ipynb
   ```

---

*Dataset: IBM Skills Network | Author: Virajbhai Mavani*
