"""
Creates SYNTHETIC test data (data/titanic_synthetic_TEST_ONLY.csv) in the Kaggle Titanic schema.
It is random data used only to test the pipeline offline. NEVER use it for your real analysis -
use the real Titanic dataset (auto-downloaded by titanic_analysis.py, or Kaggle train.csv).
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
n = 891
pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
age = np.clip(rng.normal(30, 13, n), 0.5, 80).round(0)
fare = np.where(pclass == 1, rng.gamma(3, 28, n), np.where(pclass == 2, rng.gamma(4, 5, n), rng.gamma(3, 3, n))).round(2)
p = 0.35 + 0.4 * (sex == "female") - 0.12 * (pclass - 1) - 0.004 * (age - 30)
survived = (rng.random(n) < np.clip(p, 0.03, 0.95)).astype(int)
df = pd.DataFrame({
    "PassengerId": np.arange(1, n + 1), "Survived": survived, "Pclass": pclass,
    "Name": [f"Surname{i}, {'Mr' if s == 'male' else 'Mrs'}. Test" for i, s in enumerate(sex)],
    "Sex": sex, "Age": age, "SibSp": rng.poisson(0.5, n), "Parch": rng.poisson(0.4, n),
    "Ticket": "T" + pd.Series(np.arange(n)).astype(str), "Fare": fare,
    "Cabin": np.where(rng.random(n) < 0.23, "C85", None),
    "Embarked": rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
})
df.loc[rng.choice(n, 170, replace=False), "Age"] = np.nan
df.loc[rng.choice(n, 2, replace=False), "Embarked"] = np.nan
os.makedirs("data", exist_ok=True)
df.to_csv("data/titanic_synthetic_TEST_ONLY.csv", index=False)
print("Created data/titanic_synthetic_TEST_ONLY.csv")
