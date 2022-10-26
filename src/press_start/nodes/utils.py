import importlib
from types import ModuleType
from typing import Callable, Iterable


def register_external_node(
    base_module_name: str,
    classes: Iterable[str],
    registry_module: ModuleType,
    registered_model: Callable,
):
    base_module = importlib.import_module(base_module_name)
    for class_name in classes:
        class_ = getattr(base_module, class_name)
        model = registered_model(class_)
        setattr(registry_module, class_name, model)
