import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, RFE
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def construct_target(df: pd.DataFrame) -> pd.DataFrame:
    df['CVD'] = df[['HadHeartAttack', 'HadAngina']].isin(['Yes']).any(axis=1).astype(int)
    return df

def drop_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = ['HadHeartAttack', 'HadAngina', 'HadStroke', 'HighRiskLastYear']
    return df.drop(columns=[c for c in to_drop if c in df.columns])

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    top_n = 10
    top_states = df['State'].value_counts().nlargest(top_n).index
    df['State'] = df['State'].where(df['State'].isin(top_states), 'Other')

    top_races = df['RaceEthnicityCategory'].value_counts().nlargest(5).index
    df['RaceEthnicityCategory'] = df['RaceEthnicityCategory'].where(df['RaceEthnicityCategory'].isin(top_races), 'Other')

    yes_no = [
        'HadAsthma','HadCOPD','HadDepressiveDisorder','HadKidneyDisease','HadArthritis','HadSkinCancer',
        'DifficultyWalking','DifficultyConcentrating','DifficultyDressingBathing','DifficultyErrands',
        'FluVaxLast12','HIVTesting','PneumoVaxEver','ChestScan','PhysicalActivities','DeafOrHardOfHearing',
        'BlindOrVisionDifficulty','AlcoholDrinkers'
    ]
    for col in yes_no:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1,'No':0}).fillna(0)

    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'Male':1,'Female':0}).fillna(0)

    if 'CovidPos' in df.columns:
        df['CovidPos'] = df['CovidPos'].apply(lambda x: 1 if isinstance(x, str) and ('Yes' in x or 'positive' in x) else 0)

    health_map = {'Poor':1,'Fair':2,'Good':3,'Very good':4,'Excellent':5}
    if 'GeneralHealth' in df.columns:
        df['GeneralHealth'] = df['GeneralHealth'].map(health_map).fillna(0)

    checkup_map = {
        '5 or more years ago':1,
        'Within past 5 years (2 years but less than 5 years ago)':2,
        'Within past 2 years (1 year but less than 2 years ago)':3,
        'Within past year (anytime less than 12 months ago)':4
    }
    if 'LastCheckupTime' in df.columns:
        df['LastCheckupTime'] = df['LastCheckupTime'].map(checkup_map).fillna(0)

    teeth_map = {'None of them':0,'1 to 5':1,'6 or more, but not all':2,'All':3}
    if 'RemovedTeeth' in df.columns:
        df['RemovedTeeth'] = df['RemovedTeeth'].map(teeth_map).fillna(0)

    age_map = {
       'Age 18 to 24':21,'Age 25 to 29':27,'Age 30 to 34':32,'Age 35 to 39':37,'Age 40 to 44':42,'Age 45 to 49':47,
       'Age 50 to 54':52,'Age 55 to 59':57,'Age 60 to 64':62,'Age 65 to 69':67,'Age 70 to 74':72,'Age 75 to 79':77,'Age 80 or older':85
    }
    if 'AgeCategory' in df.columns:
        df['AgeMidpoint'] = df['AgeCategory'].map(age_map).fillna(0)
        df.drop('AgeCategory', axis=1, inplace=True)

    diabetes_map = {
        'No':0,'No, pre-diabetes or borderline diabetes':1,'Yes':2,'Yes, but only during pregnancy (female)':1
    }
    if 'HadDiabetes' in df.columns:
        df['HadDiabetes'] = df['HadDiabetes'].map(diabetes_map).fillna(0)

    smoker_map = {
        'Never smoked': 0, 'Former smoker': 1, 'Current smoker - now smokes everyday': 3, 'Current smoker - now smokes some days': 2
    }
    if 'SmokerStatus' in df.columns:
        df['SmokerStatus'] = df['SmokerStatus'].map(smoker_map).fillna(0)

    ecig_map = {
        'Never used e-cigarettes in my entire life': 0, 'Not at all (right now)': 0,
        'Use them some days': 1, 'Use them everyday': 2
    }
    if 'ECigaretteUsage' in df.columns:
        df['ECigaretteUsage'] = df['ECigaretteUsage'].map(ecig_map).fillna(0)

    tetanus_map = {
        'No, did not receive any tetanus shot in the past 10 years': 0,
        'Yes, received tetanus shot but not sure what type': 1,
        'Yes, received Tdap': 2,
        'Yes, received tetanus shot, but not Tdap': 1
    }
    if 'TetanusLast10Tdap' in df.columns:
        df['TetanusLast10Tdap'] = df['TetanusLast10Tdap'].map(tetanus_map).fillna(0)

    return df

def build_preprocessor() -> ColumnTransformer:
    numeric_feats = [
        'PhysicalHealthDays','MentalHealthDays','SleepHours','BMI','HeightInMeters','WeightInKilograms','AgeMidpoint',
        'Sex','CovidPos','HadAsthma','HadCOPD','HadDepressiveDisorder','HadKidneyDisease','HadArthritis','HadSkinCancer',
        'DifficultyWalking','DifficultyConcentrating','DifficultyDressingBathing','DifficultyErrands','FluVaxLast12','HIVTesting',
        'PneumoVaxEver','ChestScan','PhysicalActivities','DeafOrHardOfHearing','BlindOrVisionDifficulty','AlcoholDrinkers',
        'HadDiabetes','SmokerStatus','ECigaretteUsage','TetanusLast10Tdap'
    ]
    ordinal_feats = ['GeneralHealth','LastCheckupTime','RemovedTeeth']
    nominal_feats = ['State','RaceEthnicityCategory']

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    ord_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('scaler', StandardScaler()),
    ])
    nom_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('ord', ord_pipe, ordinal_feats),
        ('nom', nom_pipe, nominal_feats),
    ], remainder='drop')

def build_filter_selector(k: int = 50) -> Pipeline:
    return Pipeline([
        ('var', VarianceThreshold(threshold=0.01)),
        ('mi', SelectKBest(score_func=lambda X, y: mutual_info_classif(X, y, discrete_features='auto'), k=k))
    ])

def build_wrapper_selector(estimator=None, n_features: int = 50) -> RFE:
    if estimator is None:
        estimator = LogisticRegression(class_weight='balanced', max_iter=1000)
    return RFE(estimator=estimator, n_features_to_select=n_features, step=0.1)

def build_embedded_selector() -> SelectKBest:
    lasso = LassoCV(cv=5, random_state=42)
    return SelectKBest(score_func=lambda X, y: np.abs(lasso.fit(X, y).coef_), k='all')

def prepare_data(path: str, k_filter: int = 50, k_wrapper: int = 50, test_size: float = 0.2, random_state: int = 42, selector_type: str = 'filter'):
    df = load_data(path)
    df = construct_target(df)
    df = drop_leaky_features(df)
    df = encode_categoricals(df)

    X = df.drop(columns='CVD')
    y = df['CVD']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    preprocessor = build_preprocessor()
    X_tr_p = preprocessor.fit_transform(X_tr)
    X_te_p = preprocessor.transform(X_te)

    if selector_type == 'filter':
        n_feats = X_tr_p.shape[1]
        k_use = min(k_filter, n_feats)
        selector = build_filter_selector(k=k_use)
    elif selector_type == 'wrapper':
        selector = build_wrapper_selector(n_features=k_wrapper)
    else:
        selector = build_embedded_selector()

    X_tr_s = selector.fit_transform(X_tr_p, y_tr)
    X_te_s = selector.transform(X_te_p)
    return X_tr_s, X_te_s, y_tr, y_te, preprocessor, selector

if __name__ == '__main__':
    prepare_data('../data/heart_2022_no_nans.csv')
