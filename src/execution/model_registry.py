# Routing table ( only contain static data )
DISPATCH_TABLE = {
    'svm': {
        'optuna': ('src.models.svm.svm_sklearn_optuna', 'run_svm_pipeline'),
        'pso': ('src.models.svm.svm_sklearn_pso', 'run_svm_pipeline'),
        'ga': ('src.models.svm.svm_sklearn_ga', 'run_svm_pipeline'),
        'grid': ('src.models.svm.svm_sklearn_grid', 'run_svm_pipeline')
    },
    'rf': {
        'optuna': ('src.models.rf.rf_sklearn_optuna', 'run_rf_pipeline'),
        'pso': ('src.models.rf.rf_sklearn_pso', 'run_rf_pipeline'),
        'ga': ('src.models.rf.rf_sklearn_ga', 'run_rf_pipeline'),
        'grid': ('src.models.rf.rf_sklearn_grid', 'run_rf_pipeline')
    },
    'pytorch_svm': {
        'optuna': ('src.models.svm.svm_pytorch_optuna', 'run_pytorch_pipeline')
    },
    'pytorch_mlp': {
        'optuna': ('src.models.mlp.mlp_pytorch_optuna', 'run_pytorch_mlp_pipeline')
    },
    'deepforest': {
        'optuna': ('src.models.rf.deepforest_optuna', 'run_deepforest_pipeline')
    },
    'tabnet': {
        'optuna': ('src.models.tabnet.tabnet_optuna', 'run_tabnet_pipeline')
    },
    'xgboost': {
        'custom': ('src.models.xgboost.xgboost_pipeline', 'run_xgboost_pipeline'),
        'ga':     ('src.models.xgboost.xgboost_ga_pipeline', 'run_xgboost_ga_pipeline'),
        'random': ('src.models.xgboost.xgboost_rs_pipeline', 'run_xgboost_random_pipeline'),
        'optuna': ('src.models.xgboost.xgboost_optuna_pipeline', 'run_xgboost_optuna_pipeline'),
        'grid':   ('src.models.xgboost.xgboost_grid_pipeline', 'run_xgboost_grid_pipeline'),
        'pso':    ('src.models.xgboost.xgboost_pso_pipeline', 'run_xgboost_pso_pipeline')
    },
    'decisiontree':{
        'custom': ('src.models.decisiontree.decision_tree_custom_pipeline', 'run_decision_tree_unlimited_pipeline'),
        'ga':     ('src.models.decisiontree.decision_tree_ga_pipeline', 'run_decision_tree_ga_pipeline'),
        'random': ('src.models.decisiontree.decision_tree_rs_pipeline', 'run_decision_tree_random_pipeline'),
        'optuna': ('src.models.decisiontree.decision_tree_optuna_pipeline', 'run_decision_tree_optuna_pipeline'),
        'grid':   ('src.models.decisiontree.decision_tree_grid_pipeline', 'run_decision_tree_grid_pipeline'),
        'pso':    ('src.models.decisiontree.decision_tree_pso_pipeline', 'run_decision_tree_pso_pipeline')
    },
    'logistic_regression':{
        'grid': ('src.models.logistic_reg.log_reg_pipeline', 'run_log_reg_pipeline'),
    },
    'naive_bayes':{
        'grid': ('src.models.Naive_Bayes.naive_bayes_pipeline', 'run_nb_grid_pipeline')
    }

}


def get_model_routing_info(model_name: str, optimizer: str, mode: str):
    """
    Handle exceptions (such as SGD) and return (module_path, function_name).
    """
    # SVM use SGD learning algorithm
    if model_name == 'svm' and optimizer == 'optuna' and mode == 'sgd':
        return 'src.models.svm.svm_sklearn_SGD', 'run_svm_pipeline'

    # Check if model exist
    if model_name not in DISPATCH_TABLE:
        raise ValueError(f" Model '{model_name}' not yet registered in the system.")

    routes = DISPATCH_TABLE[model_name]

    # Take optimizer pass by user
    if optimizer in routes:
        return routes[optimizer]

    # If the optimizer doesn't match, it AUTOMATICALLY takes
    # the first configuration of that model as the fallback.

    # This completely eliminates "hardcoding" the 'optuna' key.
    fallback_key = list(routes.keys())[0]
    print(
        f"[*] Warning: Automatic fallback usage '{fallback_key}' since '{optimizer}' not in the configuration of {model_name}.")

    return routes[fallback_key]