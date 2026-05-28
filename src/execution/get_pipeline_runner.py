import importlib
from src.execution.model_registry import get_model_routing_info


def get_pipeline_runner(model_name: str, optimizer: str, mode: str):
    """
    Load module động dựa trên Sổ đăng ký (Registry).
    """
    # 1. Lấy đường dẫn từ file Registry
    module_path, function_name = get_model_routing_info(model_name, optimizer, mode)

    # 2. Thực thi Import Động
    try:
        module = importlib.import_module(module_path)
        run_pipeline = getattr(module, function_name)
        return run_pipeline

    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Lỗi: Không tìm thấy file '{module_path}'.")
    except AttributeError:
        raise AttributeError(f"Lỗi: File '{module_path}' không có hàm '{function_name}'.")