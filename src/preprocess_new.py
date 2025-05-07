import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=0.3, min_samples_leaf=100):
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.encoder = TargetEncoder(
            cols=["State"],
            smoothing=self.smoothing,
            min_samples_leaf=self.min_samples_leaf
        )

    def fit(self, X, y=None):
        self.encoder = TargetEncoder(
            cols=["State"],
            smoothing=self.smoothing,
            min_samples_leaf=self.min_samples_leaf
        )
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)

def load_data(path):
    return pd.read_csv(path)

def construct_target(df):
    df['CVD'] = df[['HadHeartAttack', 'HadAngina']].isin(['Yes']).any(axis=1).astype(int)
    return df

def drop_leaky_features(df):
    to_drop = ['HadHeartAttack', 'HadAngina', 'HadStroke', 'HighRiskLastYear']
    return df.drop(columns=[c for c in to_drop if c in df.columns])

def encode_categoricals(df, y=None, is_training=True):
    if is_training:
        te = TargetEncoder(
            cols=["State"],
            smoothing=0.3,
            min_samples_leaf=100
        )
        te.fit(df["State"], y)
        encode_categoricals._state_encoder = te
    else:
        te = encode_categoricals._state_encoder

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

def build_preprocessor():
    numeric_feats = [
        'PhysicalHealthDays','MentalHealthDays','SleepHours','BMI','HeightInMeters','WeightInKilograms','AgeMidpoint',
        'Sex','CovidPos','HadAsthma','HadCOPD','HadDepressiveDisorder','HadKidneyDisease','HadArthritis','HadSkinCancer',
        'DifficultyWalking','DifficultyConcentrating','DifficultyDressingBathing','DifficultyErrands','FluVaxLast12','HIVTesting',
        'PneumoVaxEver','ChestScan','PhysicalActivities','DeafOrHardOfHearing','BlindOrVisionDifficulty','AlcoholDrinkers'
    ]
    ordinal_feats = ['GeneralHealth','LastCheckupTime','RemovedTeeth', 'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 'TetanusLast10Tdap']
    race_feat = ['RaceEthnicityCategory']
    state_feat = ['State']

    num_pipe = Pipeline([
        ('scale', StandardScaler())
    ])

    ord_pipe = Pipeline([
        ('scale', StandardScaler())
    ])

    race_pipe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    state_pipe = TargetEncoderWrapper()

    return ColumnTransformer([
        ('num',   num_pipe,   numeric_feats),
        ('ord',   ord_pipe,   ordinal_feats),
        ('race',  race_pipe,  race_feat),
        ('state', state_pipe, state_feat),
    ], remainder='drop')

def prepare_data(
    path: str,
    test_size: float = 0.2,
    random_state: int = 42
):
    df = load_data(path)
    df = construct_target(df)
    df = drop_leaky_features(df)

    X = df.drop(columns=['CVD'])
    y = df['CVD']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        stratify=y, random_state=random_state
    )

    X_tr = encode_categoricals(X_tr.copy(), y=y_tr, is_training=True)
    X_te = encode_categoricals(X_te.copy(), is_training=False)

    preprocessor = build_preprocessor()
    X_tr_p = preprocessor.fit_transform(X_tr, y_tr)
    X_te_p = preprocessor.transform(X_te)

    return X_tr_p, X_te_p, y_tr, y_te, preprocessor

if __name__ == '__main__':
    prepare_data('../data/heart_2022_no_nans.csv')