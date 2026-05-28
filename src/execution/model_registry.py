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
    }
}


# 2. HÀM TRUY XUẤT THÔNG MINH
def get_model_routing_info(model_name: str, optimizer: str, mode: str):
    """
    Xử lý các ngoại lệ (như SGD) và trả về (module_path, function_name)
    """
    # Xử lý ngoại lệ cho SVM chạy bằng SGD
    if model_name == 'svm' and optimizer == 'optuna' and mode == 'sgd':
        return 'src.models.svm.svm_sklearn_SGD', 'run_svm_pipeline'

    # Xử lý quy luật chung: Nếu model không phải svm/rf thì mặc định dùng optuna
    opt_key = optimizer if model_name in ['svm', 'rf'] else 'optuna'

    try:
        return DISPATCH_TABLE[model_name][opt_key]
    except KeyError:
        raise ValueError(f"Chưa hỗ trợ pipeline cho mô hình '{model_name}' với '{optimizer}'.")