import pandas as pd
from src.preprocessing.preprocessing import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
# XGBoost preprocessor for atp_tennis dataset
class XGBoost_preprossesor:
    def __init__(self):
        self.preprocessor=Preprocessing()
    # Preprocessing atp dataset , return processed dataset
    def preprocess(self) -> pd.DataFrame:
        data=self.preprocessor.run()
        return data
    # Split preprocessed dataset, return features and target
    def split_feat_target(self,data : pd.DataFrame):
        X=data.drop('target',axis=1)
        y=data['target']
        return X,y
    # encode str datatype in features column as category
    def encode_categorical(self, X: pd.DataFrame):
        categorical_cols = ['tourney_name', 'surface', 'tourney_level', 'round']
        for col in categorical_cols:
            X[col] = X[col].astype('category')
        return X
    # Split dataset into train,valid,test
    def split_train_val_test(self,X : pd.DataFrame,y : pd.DataFrame,test_size=0.2,val_size=0.2,random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
        return X_train,X_val,X_test,y_train,y_val,y_test
    # scale train & valid data and cast type
    def scale_train_valid(self,X_train,X_val,categorical_cols=['tourney_name', 'surface', 'tourney_level', 'round']):
        num_cols = [col for col in X_train.columns if col not in categorical_cols]
        std_scaler = StandardScaler()
        preprocessor_transformer = ColumnTransformer(
            transformers=[
                ('num', std_scaler, num_cols)
            ],
            remainder='passthrough'
        )
        all_cols = num_cols + categorical_cols
        X_train_scaled = pd.DataFrame(preprocessor_transformer.fit_transform(X_train), columns=all_cols)
        X_val_scaled = pd.DataFrame(preprocessor_transformer.transform(X_val), columns=all_cols)
        # cast type
        for col in num_cols:
            X_train_scaled[col] = X_train_scaled[col].astype(float)
            X_val_scaled[col] = X_val_scaled[col].astype(float)
        for col in categorical_cols:
            X_train_scaled[col] = X_train_scaled[col].astype('category')
            X_val_scaled[col] = X_val_scaled[col].astype('category')
        return X_train_scaled,X_val_scaled
    # preprocess for XGBoost
    def XgBoost_preprocess(self):
        data=self.preprocess()
        X,y=self.split_feat_target(data)
        X=self.encode_categorical(X)
        # note : X_test will be preprocessed before test
        X_train, X_val, X_test, y_train, y_val, y_test=self.split_train_val_test(X,y)
        X_train_scaled,X_val_scaled=self.scale_train_valid(X_train,X_val)
        return X_train_scaled,X_val_scaled,X_train,X_val,X_test,y_train,y_val,y_test
