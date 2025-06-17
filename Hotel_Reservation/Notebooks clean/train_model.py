# src/train_model.py
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle

from preprocessing import HotelDataPreprocessor
from model_training import x_train, y_train, random_search_fine

# Construction du pipeline
pipeline = Pipeline([
    ('preprocessing', HotelDataPreprocessor()),
    ('classifier', RandomForestClassifier(**random_search_fine.best_params_, random_state=42))
])

# Entraînement
pipeline.fit(x_train, y_train)

# Sauvegarde
with open('../Models/Hotel_Cancellations_Pipeline_RandomForest.pkl', 'wb') as f:
    pickle.dump(pipeline, f)