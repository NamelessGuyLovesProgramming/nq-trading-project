import os
import ast
import importlib
import sys
from pathlib import Path


def find_unused_files(project_dir):
    """Findet potenziell ungenutzte .py Dateien im Projekt."""
    # Sammle alle Python-Dateien
    all_files = list(Path(project_dir).rglob("*.py"))
    file_paths = [str(f) for f in all_files]

    # Sammle alle Importe
    imported_modules = set()
    for file_path in file_paths:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imported_modules.add(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_modules.add(node.module.split('.')[0])
        except Exception as e:
            print(f"Fehler beim Parsen von {file_path}: {e}")

    # Konvertiere Moduldateinamen
    module_files = set()
    for module in imported_modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec and spec.origin and spec.origin != "built-in":
                module_files.add(spec.origin)
        except (ImportError, ValueError):
            continue

    # Identifiziere potenziell ungenutzte Dateien
    unused_files = []

    # Exclude Liste - diese Dateien sollten nie als ungenutzt markiert werden
    exclude_list = {"main.py", "app.py", "api.py", "base_plugin.py", "__init__.py"}

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name in exclude_list:
            continue

        if file_path not in module_files and not any(
                file_path.endswith(f"/{module}.py") for module in imported_modules):
            if "__pycache__" not in file_path:
                unused_files.append(file_path)

    return unused_files


if __name__ == "__main__":
    project_dir = "."  # Aktuelles Verzeichnis
    unused_files = find_unused_files(project_dir)

    if unused_files:
        print("Potenziell unbenutzte Dateien:")
        for file in unused_files:
            print(f"  - {file}")
    else:
        print("Keine unbenutzten Dateien gefunden.")