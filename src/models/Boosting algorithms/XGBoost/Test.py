from XGBoost.XG_Boost_preprocess import XGBoost_preprossesor
from XGBoost.XGBoost_evaluate import Evaluator
from XGBoost.XGBoost_final import XGBoost_final_train
processor=XGBoost_preprossesor()
X_train_scaled,X_val_scaled,X_train,X_val,X_test,y_train,y_val,y_test=processor.XgBoost_preprocess()
evaluator=Evaluator()
n_estimators_range = [10,30, 40,50, 70, 100, 130, 160,180, 200,240, 250, 300,340,370,400,450,480,500,560,600]
best_n=evaluator.num_estimator_validation(X_train_scaled,y_train, X_val_scaled, y_val, n_estimators_range)
print(best_n)
xgb_final_train=XGBoost_final_train()
X_train_scaled,X_test_scaled,y_train,y_test=xgb_final_train.final_preprocess(X_train,y_train,X_val,y_val,X_test,y_test)
best_clf=xgb_final_train.final_train(X_train_scaled,y_train,best_n)
y_pred = best_clf.predict(X_test_scaled)
y_prob = best_clf.predict_proba(X_test_scaled)[:, 1]
evaluator.evaluate_model(y_test,y_pred,y_prob)

