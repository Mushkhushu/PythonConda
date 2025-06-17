# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../Data/Raw/Hotel Reservations.csv')
df = df.drop(columns=['booking_status'])  # cible séparée
y = pd.get_dummies(df['booking_status'], drop_first=True)['Not_Canceled']
X = df.drop(columns=['Booking_ID'])

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paramètres Random Forest
param_dist = {
    'n_estimators': np.arange(150, 301, 25),
    'max_depth': [10, 12, 14, 15, 16, 18, None],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

rf = RandomForestClassifier(random_state=42)
random_search_fine = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    verbose=2,
    random_state=42,
    n_jobs=-1
)
random_search_fine.fit(x_train, y_train)
