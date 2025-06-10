
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error


from xgboost import XGBRegressor

train_df = pd.read_csv("./train.csv", index_col="Id")
test_df = pd.read_csv("./test.csv", index_col="Id")

y = train_df["SalePrice"]
X = train_df.drop("SalePrice", axis=1)
X_test = test_df.copy()

# identify columns of type object and not object
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if X[c].dtype != "object"]

numeric_tf = SimpleImputer(strategy="mean")
# 16508.900390625 MAE
categoric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categoric_tf, cat_cols)
    ])

# pipelines
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=0,
    n_jobs=-1,
)

clf = Pipeline(steps=[
    ("pre", preprocess),
    ("model", model)
])

# split data in training data & validation data,fit and predict basic
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train,)

preds = clf.predict(X_valid)
print("Validation MAE:", mean_absolute_error(y_valid, preds))

clf.fit(X, y)

test_preds = clf.predict(X_test)

# submission
submission = pd.DataFrame({
    "Id": X_test.index,
    "SalePrice": test_preds
})
submission.to_csv("submission.csv", index=False)

print("submission.csv done✨")
