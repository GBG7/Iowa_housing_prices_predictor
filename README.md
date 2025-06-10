# 🏡 Iowa Housing Prices Predictor

My submission for Kaggle’s **IOWA Housing Prices Predictor** dataset.  
By **GBG7**

- **Submission MAE:** `14,520.93`  
  (My average guess was only off by $14,520)
- **Official tutorial MAE:** `16,520.43`

---

## 📈 XGBoost Pipeline Overview

End-to-end workflow for Kaggle’s **“House Prices: Advanced Regression Techniques”** competition.

| Stage                  | What Happens                                                                                                            | Key APIs                                  |
|------------------------|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| **1. Data split**      | Load `train.csv` / `test.csv`, extract target `SalePrice`, keep `Id` as index, create 80/20 train-validation split      | `pandas`, `train_test_split`              |
| **2. Pre-processing**  | Numeric: `SimpleImputer(mean)`<br>Categorical: `SimpleImputer(constant)` + `OneHotEncoder(handle_unknown='ignore')`     | `ColumnTransformer`, `Pipeline`           |
| **3. Model**           | Gradient-boosted trees via **`XGBRegressor`** (`n_estimators=1000`, `lr=0.05`, `n_jobs=-1`, `seed=0`)                   | `xgboost`                                 |
| **4. Train & validate**| Fit on training fold, predict validation fold, report MAE (~16,509, solid baseline, zero feature engineering)            | `mean_absolute_error`                     |
| **5. Refit & submit**  | Retrain on full data, predict for `test.csv`, output `submission.csv` for Kaggle upload                                 | `pandas`                                  |

---

## 🚀 Why This Setup?

- **One-click reproducibility:** All prep is inside a single scikit-learn pipeline (no data leak risk!).
- **Robust to messy data:** Imputes missing values and safely handles unseen categorical levels.
- **Strong baseline:** XGBoost usually lands in the LB’s top half with minimal tuning; hyper-params are easy to grid-search later.

---

## 🛠️ Run Locally

```bash
pip install -r requirements.txt
python Iowa_house_prices_predictor.py
# => submission.csv ✨
