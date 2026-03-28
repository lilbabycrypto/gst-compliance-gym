# Root conftest.py — ensures pytest does not try to import the root __init__.py
# as a test module. The root __init__.py uses relative imports that are only
# valid when the package is installed, not when run standalone by pytest.
collect_ignore = ["__init__.py", "client.py", "models.py"]
