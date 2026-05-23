import pandas as pd
from XG_Boost_preprocess import XGBoost_preprossesor
from xgboost import XGBClassifier
class XGBoost_final_train:
    def final_preprocess(self,X_train,y_train,X_val,y_val,X_test,y_test):
        X_train = pd.concat([X_train, X_val])
        y_train = pd.concat([y_train,y_val])
        processor=XGBoost_preprossesor()
        X_train_scaled,X_test_scaled=processor.scale_train_valid(X_train,X_test)
        return X_train_scaled,X_test_scaled,y_train,y_test
    def final_train(self,X_train_scaled,y_train,best_n):
        clf = XGBClassifier(
            n_estimators=best_n,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
            eval_metric='logloss',
            enable_categorical=True,
            tree_method='hist'
        )
        clf.fit(X_train_scaled,y_train)
        return clf
