import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

class HotelDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = None
        self.prix_moyen_par_type = {}
        self.feature_names = None

    def fit(self, X, y=None):
        df = X.copy()

        if 'Booking_ID' in df.columns:
            df = df.drop(['Booking_ID'], axis=1)

        # 2. One-Hot Encoding
        self.encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
        categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']
        self.encoder.fit(df[categorical_cols])

        # Calculer les prix moyens par type de chambre après encoding
        cols_encoded = self.encoder.transform(df[categorical_cols])
        labels = self.encoder.get_feature_names_out(categorical_cols)
        df_encoded = pd.DataFrame(cols_encoded, columns=labels, index=df.index)
        df_temp = df.join(df_encoded.astype(int))

        # Calculer prix moyens par type de chambre
        room_type_cols = [col for col in labels if col.startswith('room_type_reserved_')]
        for col in room_type_cols:
            mask = (df_temp[col] == 1) & (df_temp['avg_price_per_room'] > 0)
            if mask.sum() > 0:
                prix_moyen = df_temp.loc[mask, 'avg_price_per_room'].mean()
                self.prix_moyen_par_type[col] = prix_moyen

        return self

    def transform(self, X):
        df = X.copy()

        if 'Booking_ID' in df.columns:
            df = df.drop(['Booking_ID'], axis=1)

        # 2. One-Hot Encoding
        base_categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']
        categorical_cols = [col for col in base_categorical_cols if col in df.columns]

        # Si booking_status n'est pas présent, on doit l'ajouter temporairement pour l'encoder
        temp_booking_status_added = False
        if 'booking_status' not in df.columns:
            df['booking_status'] = 'Not_Canceled'  # Valeur par défaut
            categorical_cols.append('booking_status')
            temp_booking_status_added = True

        cols_encoded = self.encoder.transform(df[categorical_cols])
        labels = self.encoder.get_feature_names_out(categorical_cols)
        df_encoded = pd.DataFrame(cols_encoded, columns=labels, index=df.index)
        df = df.join(df_encoded.astype(int))
        df = df.drop(categorical_cols, axis=1)

        # Si on a ajouté booking_status temporairement, on supprime les colonnes correspondantes
        if temp_booking_status_added:
            booking_status_cols = [col for col in df.columns if col.startswith('booking_status_')]
            df = df.drop(columns=booking_status_cols)

        # 3. Remplacer les prix moyens à 0
        room_type_cols = [col for col in df.columns if col.startswith('room_type_reserved_')]

        def remplacer_prix_moyen(row):
            if row['avg_price_per_room'] == 0:
                for col in room_type_cols:
                    if row[col] == 1:
                        return self.prix_moyen_par_type.get(col, np.nan)
                return np.nan
            else:
                return row['avg_price_per_room']

        df['avg_price_per_room'] = df.apply(remplacer_prix_moyen, axis=1)

        # 4. Simplifier les meal plans
        meal_cols = [col for col in df.columns if col.startswith('type_of_meal_plan_') and 'Not Selected' not in col]
        df['meal_plan_selected'] = df[meal_cols].sum(axis=1).apply(lambda x: 1 if x > 0 else 0)

        # Supprimer les colonnes meal plan originales
        cols_to_drop = [col for col in df.columns if col.startswith('type_of_meal_plan_')]
        df = df.drop(columns=cols_to_drop)

        # 5. Convertir room_type en numérique
        room_type_cols = [col for col in df.columns if col.startswith('room_type_reserved_')]
        df['room_type_reserved'] = df[room_type_cols].idxmax(axis=1).str.extract(r'Room_Type (\d)').astype(int)
        df = df.drop(columns=room_type_cols)

        # 6. Créer les features engineerées
        df['total_nights'] = df['no_of_week_nights'] + df['no_of_weekend_nights']
        df['total_people'] = df['no_of_adults'] + df['no_of_children']

        # Price per person avec gestion des divisions par zéro
        df['price_per_person'] = df['avg_price_per_room'] / df['total_people']
        df['price_per_person'] = df['price_per_person'].replace([np.inf, -np.inf], np.nan)
        df['price_per_person'] = df['price_per_person'].fillna(df['avg_price_per_room'])

        df['is_family'] = (df['no_of_children'] > 0).astype(int)
        df['stay_duration_flag'] = (df['total_nights'] > 7).astype(int)

        # 7. Lead time categories
        df['lead_time_category'] = pd.cut(
            df['lead_time'],
            bins=[-1, 7, 30, 90, 180, 443],
            labels=['0-7j', '8-30j', '31-90j', '91-180j', '181j+']
        )
        df = pd.get_dummies(df, columns=['lead_time_category'], drop_first=True)

        # Convertir les colonnes lead_time en int
        lead_time_cols = [col for col in df.columns if col.startswith('lead_time_category_')]
        df[lead_time_cols] = df[lead_time_cols].astype(int)

        # 8. Sélectionner les features finales
        features = [
            'no_of_special_requests', 'price_per_person',
            'required_car_parking_space', 'total_nights', 'total_people',
            'room_type_reserved', 'repeated_guest', 'no_of_previous_cancellations',
            'meal_plan_selected', 'avg_price_per_room',
            'market_segment_type_Online', 'is_family', 'stay_duration_flag'
        ]

        # Ajouter les colonnes lead_time qui existent
        for col in lead_time_cols:
            if col in df.columns:
                features.append(col)

        # Vérifier que toutes les features existent
        existing_features = [f for f in features if f in df.columns]

        return df[existing_features]


# Fonction pour créer et entraîner le pipeline complet
def create_complete_pipeline():
    # Charger les données
    df = pd.read_csv('../Data/Raw/Hotel Reservations.csv')

    # Vérifier que booking_status existe pour l'entraînement
    if 'booking_status' not in df.columns:
        raise ValueError("La colonne 'booking_status' est nécessaire pour l'entraînement")

    # Séparer features et target
    X = df.drop(['booking_status'], axis=1)

    # Le target sera extrait par le preprocessor puis supprimé
    preprocessor = HotelDataPreprocessor()

    # Pour l'entraînement, on a besoin du target
    df_temp = df.copy()
    if 'Booking_ID' in df_temp.columns:
        df_temp = df_temp.drop(['Booking_ID'], axis=1)

    # One-hot encode pour extraire le target
    encoder_temp = OneHotEncoder(drop='if_binary', sparse_output=False)
    categorical_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']
    cols_encoded = encoder_temp.fit_transform(df_temp[categorical_cols])
    labels = encoder_temp.get_feature_names_out(categorical_cols)
    df_encoded = pd.DataFrame(cols_encoded, columns=labels, index=df_temp.index)

    # Le target est booking_status_Not_Canceled
    y = df_encoded['booking_status_Not_Canceled'].astype(int)

    # Créer le pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Paramètres pour RandomizedSearch
    param_dist = {
        'classifier__n_estimators': np.arange(150, 301, 25),
        'classifier__max_depth': [10, 12, 14, 15, 16, 18, None],
        'classifier__min_samples_split': [2, 3, 4, 5],
        'classifier__min_samples_leaf': [1, 2, 3],
        'classifier__max_features': ['sqrt', 'log2', None],
        'classifier__bootstrap': [True, False]
    }

    # RandomizedSearchCV sur le pipeline complet
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='accuracy',
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
    # Entraîner
    random_search.fit(X_train, y_train)  # On passe le DataFrame original

    return random_search


# Créer et entraîner le pipeline
complete_pipeline = create_complete_pipeline()

# Sauvegarder le pipeline complet
with open('../Models/Hotel_Complete_Pipeline.pkl', 'wb') as file:
    pickle.dump(complete_pipeline, file)

# Exemple d'utilisation du pipeline sauvegardé
def predict_with_pipeline(data_path, pipeline_path):
    """
    Fonction pour faire des prédictions avec le pipeline sauvegardé
    Les données de test ne doivent PAS contenir la colonne 'booking_status'
    """
    # Charger le pipeline
    with open(pipeline_path, 'rb') as file:
        pipeline = pickle.load(file)

    # Charger nouvelles données
    new_data = pd.read_csv(data_path)

    # Vérifier que booking_status n'est pas présent (c'est ce qu'on veut prédire)
    if 'booking_status' in new_data.columns:
        new_data = new_data.drop(['booking_status'], axis=1)

    # Faire les prédictions
    predictions = pipeline.predict(new_data)
    probabilities = pipeline.predict_proba(new_data)

    return predictions, probabilities