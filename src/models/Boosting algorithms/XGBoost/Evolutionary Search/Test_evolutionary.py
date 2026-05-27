from Evolutionary_tuning import XGBGeneticSearch
from XGBoost.XGBoost_evaluate import Evaluator
from XGBoost.XG_Boost_preprocess import XGBoost_preprossesor
from sklearn.model_selection import train_test_split

processor = XGBoost_preprossesor()
X,y=processor.split_feat_target(processor.preprocess())
X=processor.encode_categorical(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train_scaled,X_test_scaled=processor.scale_train_valid(X_train,X_test)
searcher = XGBGeneticSearch()
best_clf = searcher.search(X_train_scaled, y_train)
y_pred = best_clf.predict(X_test_scaled)
y_prob = best_clf.predict_proba(X_test_scaled)[:, 1]
evaluator = Evaluator()
evaluator.evaluate_model(y_test, y_pred, y_prob)