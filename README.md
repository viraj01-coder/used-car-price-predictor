# Used Car Price Predictor

A machine learning project that predicts used car prices using Ridge Regression
with Polynomial Features. The goal is to identify key factors affecting car prices
and build an accurate, deployable prediction model.

---

## Live Demo

Check out the interactive Streamlit app:

**[Launch App](https://used-car-price-predictor-6g2wersaihrvzufyy7xjne.streamlit.app/)**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Live Demo](#live-demo)
- [Dataset](#dataset)
- [Tools & Libraries](#tools--libraries)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Model Performance](#model-performance)
- [How to Run Locally](#how-to-run-locally)

---

## Project Overview

This project performs end-to-end analysis on a UK used car dataset.
It covers data cleaning, exploratory data analysis, feature correlation,
machine learning model training with hyperparameter tuning via GridSearchCV,
and a live Streamlit web app with real-time price prediction and multi-currency support.

---

## Dataset

- **Source:** IBM Skills Network (Used Car Price Analysis)
- **Records:** ~18,000 cars
- **Domain:** UK Used Car Market (Ford models)
- **Features:** Year, Mileage, Fuel Type, Transmission, Tax, MPG, Engine Size, Price

---

## Tools & Libraries

| Tool / Library | Purpose |
|----------------|---------|
| Python 3 | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Plotly Express | Interactive visualizations |
| scikit-learn | Machine learning model training |
| Streamlit | Interactive web app |
| Jupyter Notebook | Analysis and exploration |

---

## Project Structure

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

## Key Findings

1. **Year of manufacture** is the strongest price predictor with correlation of 0.636
2. **Mileage** is the second most important factor — higher mileage = lower price (0.531)
3. **Engine size** has a positive impact on price — larger engines cost more (0.411)
4. **Tax** also correlates with pricing patterns (0.406)
5. **Polynomial Ridge Regression** (degree=2) significantly outperforms simple linear models
6. **Optimal alpha = 100** found via GridSearchCV with 4-fold cross-validation

---

## Model Performance

| Model | R² Score |
|-------|----------|
| Ridge Regression (α=0.1) | ~0.68 |
| Polynomial Ridge (degree=2, α=0.1) | 0.7557 |
| Polynomial Ridge (degree=2, α=100) ✅ | **0.7560** |

---

## How to Run Locally

1. Clone this repository
   ```
   git clone https://github.com/your-username/used-car-price-predictor.git
   ```

2. Install required libraries
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit app
   ```
   streamlit run app.py
   ```

4. Open the notebook
   ```
   jupyter notebook car_price.ipynb
   ```

---

*Dataset: IBM Skills Network | Author: Virajbhai Mavani*
