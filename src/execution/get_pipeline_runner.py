import importlib

from src.execution.model_registry import get_model_routing_info


def get_pipeline_runner(model_name: str, optimizer: str, mode: str):
    """
    Load module based on registry.
    """
    # Take path from file Registry
    module_path, function_name = get_model_routing_info(model_name, optimizer, mode)

    # dynamic import
    try:
        module = importlib.import_module(module_path)
        run_pipeline = getattr(module, function_name)
        return run_pipeline

    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Error: Not found file '{module_path}'.")
    except AttributeError:
        raise AttributeError(f"Error: File '{module_path}' do not have '{function_name}'.")