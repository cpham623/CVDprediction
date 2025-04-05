import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def construct_target(df):
    # Still not sure if this is a good target
    df['CVD'] = df[['HadHeartAttack', 'HadStroke', 'HadAngina']].apply(
        lambda x: 1 if 'Yes' in x.values else 0, axis=1
    )
    return df


def encode_categoricals(df):
    # Binary Yes/No columns
    binary_cols = [
        'HadHeartAttack', 'HadStroke', 'HadAngina', 'HadAsthma', 'HadCOPD',
        'DeafOrHardOfHearing', 'DifficultyWalking', 'FluVaxLast12', 'HIVTesting',
        'HadDepressiveDisorder', 'DifficultyErrands', 'DifficultyDressingBathing',
        'DifficultyConcentrating', 'AlcoholDrinkers', 'ChestScan', 'HighRiskLastYear',
        'HadSkinCancer', 'HadArthritis', 'HadKidneyDisease', 'BlindOrVisionDifficulty',
        'PhysicalActivities', 'PneumoVaxEver', 'Sex'
    ]

    for col in binary_cols:
        df[col] = df[col].map({'No': 0, 'Yes': 1})

    # CovidPos special case
    df['CovidPos'] = df['CovidPos'].apply(lambda x: 1 if 'Yes' in str(x) or 'positive' in str(x) else 0)

    # Ordinal mappings
    health_map = {
        'Excellent': 5, 'Very good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1
    }
    df['GeneralHealth'] = df['GeneralHealth'].map(health_map)

    checkup_map = {
        'Within past year (anytime less than 12 months ago)': 4,
        'Within past 2 years (1 year but less than 2 years ago)': 3,
        'Within past 5 years (2 years but less than 5 years ago)': 2,
        '5 or more years ago': 1
    }
    df['LastCheckupTime'] = df['LastCheckupTime'].map(checkup_map)

    teeth_map = {
        'None of them': 0, '1 to 5': 1, '6 or more, but not all': 2, 'All': 3
    }
    df['RemovedTeeth'] = df['RemovedTeeth'].map(teeth_map)

    age_map = {
        'Age 18 to 24': 21, 'Age 25 to 29': 27, 'Age 30 to 34': 32,
        'Age 35 to 39': 37, 'Age 40 to 44': 42, 'Age 45 to 49': 47,
        'Age 50 to 54': 52, 'Age 55 to 59': 57, 'Age 60 to 64': 62,
        'Age 65 to 69': 67, 'Age 70 to 74': 72, 'Age 75 to 79': 77,
        'Age 80 or older': 85
    }
    df['AgeMidpoint'] = df['AgeCategory'].map(age_map)
    df.drop('AgeCategory', axis=1, inplace=True)

    diabetes_map = {
        'No': 0,
        'No, pre-diabetes or borderline diabetes': 1,
        'Yes': 2,
        'Yes, but only during pregnancy (female)': 1
    }
    df['HadDiabetes'] = df['HadDiabetes'].map(diabetes_map)

    # Nominal variables for one-hot encoding
    nominal_cols = [
        'State', 'RaceEthnicityCategory', 'SmokerStatus', 'ECigaretteUsage', 'TetanusLast10Tdap'
    ]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    return df


def scale_features(df):
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include='number').columns.drop('CVD')
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def split_data(df):
    X = df.drop('CVD', axis=1)
    y = df['CVD']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(path):
    df = load_data(path)
    df = construct_target(df)
    df = encode_categoricals(df)
    df = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test

