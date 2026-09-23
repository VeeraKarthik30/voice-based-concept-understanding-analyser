import importlib


def test_app_module_imports():
    app_module = importlib.import_module("app")
    assert hasattr(app_module, "main")
