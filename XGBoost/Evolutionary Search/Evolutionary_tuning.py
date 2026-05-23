from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Continuous
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold


class XGBGeneticSearch:
    def __init__(self,generations=40,tournament_size=3,population_size=30):
        # Classifier
        self.clf = XGBClassifier(
            enable_categorical=True,
            tree_method='hist',
            eval_metric='logloss',
            random_state=42
        )
        # Search space
        self.param_grid = {
            'n_estimators': Integer(100, 800),
            'max_depth': Integer(3, 8),
            'learning_rate': Continuous(0.01, 0.2, distribution='log-uniform'),
            'subsample': Continuous(0.6, 0.9),
            'colsample_bytree': Continuous(0.6, 0.9),
            'gamma': Continuous(0, 2)
        }
        # genetic estimator
        self.generations=generations
        self.population_size=population_size
        self.tournament_size=tournament_size

        self.evolved_estimator = GASearchCV(
            estimator=self.clf,
            cv=StratifiedKFold(n_splits=3),
            scoring='neg_log_loss',
            param_grid=self.param_grid,
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=self.tournament_size,
            elitism=True,
            verbose=True,
            n_jobs=-1
        )

    def search(self, X_train_scaled, y_train):
        self.evolved_estimator.fit(X_train_scaled, y_train)
        print(f"Best parameters: {self.evolved_estimator.best_params_}")
        return self.evolved_estimator.best_estimator_