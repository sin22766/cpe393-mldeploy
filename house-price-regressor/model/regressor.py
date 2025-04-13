import os
import pickle
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

current_dir = os.path.dirname(os.path.abspath(__file__))
housing_path = os.path.join(current_dir, "Housing.csv")

housing = pd.read_csv(housing_path)

# Transform yes/no to bool
bool_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]
for col in bool_columns:
    housing[col] = housing[col].map({"yes": True, "no": False})
housing["furnishingstatus"] = housing["furnishingstatus"].map(
    {"furnished": 1, "semi-furnished": 0.5, "unfurnished": 0}
)

X = housing.drop("price", axis=1)
y = housing["price"]

params = {
    "loss": "squared_error",
    "max_features": "sqrt",
    "learning_rate": 0.03903713067765278,
    "n_estimators": 385,
    "subsample": 0.9307711606738052,
    "max_depth": 4,
    "min_samples_split": 11,
    "min_samples_leaf": 1,
    "alpha": 0.9012805033810811,
    "random_state": 42,
}

model = GradientBoostingRegressor(**params)
model.fit(X, y)

with open("app/model.pkl", "wb") as f:
    pickle.dump(model, f)

