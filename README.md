My submission for the Kaggle's IOWA Housing Prices Predictor dataset.
By GBG7

Submission score got me a MAE (Mean Absolute Error) of 14520.93. 
This means that my average guess was only off by $14,520. 

Compare it to the official tutorial's MAE of 16520.432.

Write up:
## 🏡 House Prices — XGBoost Pipeline

End-to-end workflow for Kaggle’s **“House Prices: Advanced Regression Techniques”** competition.

| Stage | What happens | Key APIs |
|-------|--------------|----------|
| **1&nbsp;· Data split** | Load `train.csv` / `test.csv`, extract target **`SalePrice`**, keep **`Id`** as index, create an 80 / 20 train-validation split for honest scoring. | `pandas`, `train_test_split` |
| **2&nbsp;· Pre-processing** | *Numeric* columns → `SimpleImputer(mean)`  <br>*Categorical* columns → `SimpleImputer(constant)` + `OneHotEncoder(ignore_unknown)` | `ColumnTransformer`, `Pipeline` |
| **3&nbsp;· Model** | Gradient-boosted trees via **`XGBRegressor`** (`n_estimators=1000`, `lr=0.05`, `n_jobs=-1`, `seed=0`) | `xgboost` |
| **4&nbsp;· Train + validate** | Fit on the training fold, predict the validation fold, report **MAE ≈ 16 509** (solid baseline, zero feature engineering). | `mean_absolute_error` |
| **5&nbsp;· Refit + submit** | Re-train on the full dataset, generate predictions for `test.csv`, write **`submission.csv`** ready for Kaggle upload. | `pandas` |

### Why this setup?

* **One-click reproducibility** – all prep is inside a single scikit-learn pipeline → no data-leak risk.  
* **Robust to messy data** – imputes missing values and safely handles unseen categorical levels.  
* **Strong baseline** – XGBoost usually lands on the LB’s top half with minimal tuning; hyper-params are easy to grid-search later.

---

Run locally:

```bash
pip install -r requirements.txt
Iowa_house_prices_predictor.py
# => submission.csv ✨
```

