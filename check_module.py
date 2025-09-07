#!/usr/bin/env python3
"""
Diagnostic script to check what's available in dataset_generator module.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"Python version: {sys.version}")
print(f"Current directory: {Path.cwd()}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
print("-" * 50)

try:
    import dataset_generator

    print("✅ Successfully imported dataset_generator module")

    print(f"Module file: {getattr(dataset_generator, '__file__', 'Unknown')}")

    # Get all attributes
    all_attrs = dir(dataset_generator)
    public_attrs = [attr for attr in all_attrs if not attr.startswith("_")]
    classes = []
    functions = []
    variables = []

    for attr_name in public_attrs:
        attr = getattr(dataset_generator, attr_name)
        if hasattr(attr, "__bases__"):  # It's a class
            classes.append(attr_name)
        elif callable(attr):
            functions.append(attr_name)
        else:
            variables.append(attr_name)

    print(f"\n📋 Module contents:")
    print(f"   Classes: {classes}")
    print(f"   Functions: {functions}")
    print(f"   Variables: {variables}")

    # Try to find what we're looking for
    target_classes = [
        "RequirementsDatasetGenerator",
        "DatasetGenerator",
        "Generator",
        "MultimediaDatasetGenerator",
    ]

    found_targets = []
    for target in target_classes:
        if hasattr(dataset_generator, target):
            found_targets.append(target)
            target_class = getattr(dataset_generator, target)
            print(f"\n✅ Found target class: {target}")

            # Check methods of the class
            if hasattr(target_class, "__init__"):
                try:
                    instance = target_class()
                    methods = [
                        method
                        for method in dir(instance)
                        if not method.startswith("_")
                        and callable(getattr(instance, method))
                    ]
                    print(
                        f"   Available methods: {methods[:10]}{'...' if len(methods) > 10 else ''}"
                    )
                except Exception as e:
                    print(f"   Could not instantiate: {e}")

    if not found_targets:
        print(f"\n❌ No target classes found. Looking for: {target_classes}")
        if classes:
            print(f"   But found these classes: {classes}")
            # Try to use the first available class
            first_class_name = classes[0]
            first_class = getattr(dataset_generator, first_class_name)
            print(f"   Trying to use {first_class_name} instead...")
            try:
                instance = first_class()
                methods = [
                    method
                    for method in dir(instance)
                    if not method.startswith("_")
                    and callable(getattr(instance, method))
                ]
                print(f"   Methods available: {methods}")
            except Exception as e:
                print(f"   Failed to instantiate {first_class_name}: {e}")

except ImportError as e:
    print(f"❌ Failed to import dataset_generator: {e}")

    # Check if file exists
    dataset_gen_file = project_root / "dataset_generator.py"
    if dataset_gen_file.exists():
        print(f"✅ File exists at: {dataset_gen_file}")

        # Try to read first few lines
        try:
            with open(dataset_gen_file, "r", encoding="utf-8") as f:
                first_lines = [f.readline().strip() for _ in range(10)]
            print(f"\n📄 First 10 lines of file:")
            for i, line in enumerate(first_lines, 1):
                if line:
                    print(f"   {i:2}: {line}")
        except Exception as e:
            print(f"❌ Could not read file: {e}")
    else:
        print(f"❌ File does not exist at: {dataset_gen_file}")

print("\n" + "=" * 50)
print("Run this script to diagnose your module issues:")
print("python check_module.py")
