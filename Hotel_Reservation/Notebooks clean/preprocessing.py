from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np



class HotelDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # Pas besoin d'apprendre quoi que ce soit ici

    def transform(self, X):
        df_clean = X.copy()

        # Suppression de colonnes inutiles
        if 'Booking_ID' in df_clean.columns:
            df_clean = df_clean.drop(['Booking_ID'], axis=1)

        # Encodage one-hot de certaines colonnes
        encoder = OneHotEncoder(drop='if_binary', sparse=False)
        encoded = encoder.fit_transform(df_clean[['type_of_meal_plan','room_type_reserved', 'market_segment_type','booking_status']])
        labels = encoder.get_feature_names_out(['type_of_meal_plan','room_type_reserved', 'market_segment_type','booking_status'])
        df_encoded = pd.DataFrame(encoded, columns=labels, index=df_clean.index)
        df_clean = df_clean.drop(['type_of_meal_plan','room_type_reserved', 'market_segment_type','booking_status'], axis=1)
        df_clean = pd.concat([df_clean, df_encoded], axis=1)

        # Imputation avg_price_per_room selon type de chambre
        room_type_cols = [col for col in df_encoded.columns if col.startswith('room_type_reserved')]
        prix_moyen_par_type = {}
        for col in room_type_cols:
            prix_moyen = df_clean.loc[(df_clean[col] == 1) & (df_clean['avg_price_per_room'] > 0), 'avg_price_per_room'].mean()
            prix_moyen_par_type[col] = prix_moyen

        def remplacer(row):
            if row['avg_price_per_room'] == 0:
                for col in room_type_cols:
                    if row[col] == 1:
                        return prix_moyen_par_type.get(col, np.nan)
            return row['avg_price_per_room']

        df_clean['avg_price_per_room'] = df_clean.apply(remplacer, axis=1)

        # meal_plan_selected
        meal_cols = [col for col in df_encoded.columns if col.startswith('type_of_meal_plan_') and col != 'type_of_meal_plan_Not Selected']
        df_clean['meal_plan_selected'] = df_clean[meal_cols].sum(axis=1)
        df_clean['meal_plan_selected'] = (df_clean['meal_plan_selected'] > 0).astype(int)
        df_clean = df_clean.drop(meal_cols + ['type_of_meal_plan_Not Selected'], axis=1, errors='ignore')

        # room_type_reserved compressée
        df_clean['room_type_reserved'] = df_encoded[room_type_cols].idxmax(axis=1).str.extract(r'Room_Type (\d)').astype(int)
        df_clean = df_clean.drop(columns=room_type_cols)

        # Colonnes dérivées
        df_clean['total_nights'] = df_clean['no_of_week_nights'] + df_clean['no_of_weekend_nights']
        df_clean['total_people'] = df_clean['no_of_adults'] + df_clean['no_of_children']
        df_clean['price_per_person'] = df_clean['avg_price_per_room'] / df_clean['total_people']
        df_clean['price_per_person'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_clean['price_per_person'].fillna(df_clean['avg_price_per_room'], inplace=True)
        df_clean['is_family'] = (df_clean['no_of_children'] > 0).astype(int)
        df_clean['stay_duration_flag'] = (df_clean['total_nights'] > 7).astype(int)

        df_clean['lead_time_category'] = pd.cut(
            df_clean['lead_time'],
            bins=[-1, 7, 30, 90, 180, 443],
            labels=['0-7j', '8-30j', '31-90j', '91-180j', '181j+']
        )
        df_clean = pd.get_dummies(df_clean, columns=['lead_time_category'], drop_first=True)

        for col in ['lead_time_category_8-30j', 'lead_time_category_31-90j',
                    'lead_time_category_91-180j', 'lead_time_category_181j+']:
            if col not in df_clean.columns:
                df_clean[col] = 0
            df_clean[col] = df_clean[col].astype(int)

        return df_clean
