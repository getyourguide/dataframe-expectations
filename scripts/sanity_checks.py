"""
DataFrame Expectations Framework Sanity Check Script

This script validates consistency across the entire expectations framework by checking:
1. All expectations implemented in the expectations/ directory are registered in the registry
2. All registered expectations have corresponding expect_* methods in DataFrameExpectationsSuite
3. All registered expectations have corresponding unit tests in tests/expectations/

Usage:
    python scripts/sanity_checks.py
    python scripts/sanity_checks.py --verbose
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, Optional


class ExpectationsSanityChecker:
    """Validates consistency across the expectations framework."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.expectations_dir = project_root / "dataframe_expectations" / "expectations"
        self.stub_file = project_root / "dataframe_expectations" / "suite.pyi"
        self.tests_dir = project_root / "tests" / "expectations"

        # Results storage
        self.registered_expectations: Dict[str, str] = {}  # expectation_name -> file_path
        self.suite_methods: set[str] = set()  # expect_* method names
        self.test_files: Dict[str, str] = {}  # expectation_name -> test_file_path
        self.issues: list[str] = []

    def run_full_check(self) -> bool:
        """Run all consistency checks and return True if all pass."""
        print("🔍 Starting DataFrame Expectations Framework Sanity Check...")
        print("=" * 70)

        steps = [
            ("📋 Discovering registered expectations", self._discover_registered_expectations,
             lambda: f"Found {len(self.registered_expectations)} registered expectations"),
            ("🎯 Discovering suite methods", self._discover_suite_methods,
             lambda: f"Found {len(self.suite_methods)} expect_* methods"),
            ("🧪 Discovering test files", self._discover_test_files,
             lambda: f"Found {len(self.test_files)} test files"),
            ("📝 Validating stub file", self._validate_stub_file, None),
            ("✅ Validating consistency", self._validate_consistency, None),
            ("🏷️  Checking expectation constructors", self._check_expectation_constructor_tags, None),
        ]

        for i, (description, func, result_msg) in enumerate(steps, 1):
            print(f"\nStep {i}: {description}...")
            func()
            if result_msg:
                print(f"   {result_msg()}")

        self._print_results()
        return len(self.issues) == 0

    def _validate_consistency(self):
        """Run all consistency validation checks."""
        self._validate_registry_to_suite_mapping()
        self._validate_registry_to_tests_mapping()
        self._validate_orphaned_suite_methods()
        self._validate_orphaned_test_files()

    def _discover_registered_expectations(self):
        """Find all @register_expectation decorators in expectation files."""
        for file_path in self.expectations_dir.rglob("*.py"):
            if file_path.name == "__init__.py":
                continue

            try:
                tree = ast.parse(file_path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        expectation_name = self._extract_registered_expectation_name(node)
                        if expectation_name:
                            self.registered_expectations[expectation_name] = str(file_path)
            except Exception as e:
                print(f"   ⚠️  Warning: Could not parse {file_path.name}: {e}")

    def _extract_registered_expectation_name(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Extract expectation name from @register_expectation decorator if present."""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name) and decorator.func.id == "register_expectation":
                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                        return str(decorator.args[0].value)
        return None

    def _discover_suite_methods(self):
        """Find all expect_* methods available via the registry."""
        try:
            from dataframe_expectations.registry import DataFrameExpectationRegistry
            self.suite_methods = set(DataFrameExpectationRegistry.get_suite_method_mapping().keys())
        except Exception as e:
            self.issues.append(f"❌ Could not load suite methods from registry: {e}")

    def _discover_test_files(self):
        """Find all test files and map them to expectation names."""
        if not self.tests_dir.exists():
            self.issues.append(f"❌ Tests directory not found: {self.tests_dir}")
            return

        for test_file in self.tests_dir.rglob("test_*.py"):
            if "template" in test_file.name.lower():
                continue

            # Convert test_expect_value_equals.py -> ExpectationValueEquals
            if test_file.stem.startswith("test_expect_"):
                expectation_part = test_file.stem[12:]  # Remove "test_expect_"
                expectation_name = f"Expectation{self._snake_to_pascal_case(expectation_part)}"
                self.test_files[expectation_name] = str(test_file)

    def _validate_stub_file(self):
        """Check if the stub file is up-to-date with registered expectations."""
        if not self.stub_file.exists():
            self.issues.append(
                f"❌ Stub file not found: {self.stub_file}\n"
                "     Run: python scripts/generate_suite_stubs.py"
            )
            return

        try:
            from generate_suite_stubs import generate_pyi_file

            expected_content = generate_pyi_file()
            actual_content = self.stub_file.read_text()

            if expected_content != actual_content:
                self.issues.append(
                    "❌ Stub file is out of date\n"
                    "     Run: python scripts/generate_suite_stubs.py"
                )
        except Exception as e:
            self.issues.append(f"❌ Could not validate stub file: {e}")

    @staticmethod
    def _snake_to_pascal_case(snake_str: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in snake_str.split("_"))

    def _validate_registry_to_suite_mapping(self):
        """Check that all registered expectations have suite methods."""
        missing = [
            (name, self._expectation_to_suite_method(name))
            for name in self.registered_expectations
            if self._expectation_to_suite_method(name) not in self.suite_methods
        ]

        if missing:
            self.issues.append("❌ Registered expectations missing suite methods:")
            self.issues.extend(f"     • {name} -> missing {method}()" for name, method in missing)

    def _validate_registry_to_tests_mapping(self):
        """Check that all registered expectations have test files."""
        missing = [name for name in self.registered_expectations if name not in self.test_files]

        if missing:
            self.issues.append("❌ Registered expectations missing test files:")
            self.issues.extend(
                f"     • {name} -> missing {self._expectation_to_test_filename(name)}"
                for name in missing
            )

    def _validate_orphaned_suite_methods(self):
        """Check for suite methods without corresponding registered expectations."""
        orphaned = [
            (method, self._suite_method_to_expectation(method))
            for method in self.suite_methods
            if self._suite_method_to_expectation(method) not in self.registered_expectations
        ]

        if orphaned:
            self.issues.append("❌ Suite methods without registered expectations:")
            self.issues.extend(f"     • {method}() -> missing {exp}" for method, exp in orphaned)

    def _validate_orphaned_test_files(self):
        """Check for test files without corresponding registered expectations."""
        orphaned = [
            (name, path)
            for name, path in self.test_files.items()
            if name not in self.registered_expectations
        ]

        if orphaned:
            self.issues.append("❌ Test files without registered expectations:")
            self.issues.extend(f"     • {path} -> missing {name}" for name, path in orphaned)

    @staticmethod
    def _expectation_to_suite_method(expectation_name: str) -> str:
        """Convert ExpectationFooBar to expect_foo_bar."""
        name_part = expectation_name.removeprefix("Expectation")
        snake_case = re.sub("([A-Z])", r"_\1", name_part).lower().lstrip("_")
        return f"expect_{snake_case}"

    def _suite_method_to_expectation(self, method_name: str) -> str:
        """Convert expect_foo_bar to ExpectationFooBar."""
        if method_name.startswith("expect_"):
            name_part = method_name[7:]
            return f"Expectation{self._snake_to_pascal_case(name_part)}"
        return method_name

    def _expectation_to_test_filename(self, expectation_name: str) -> str:
        """Convert expectation name to test_expect_foo_bar.py."""
        return f"test_{self._expectation_to_suite_method(expectation_name)}.py"

    def _check_expectation_constructor_tags(self):
        """Check that all DataFrameExpectation subclasses accept 'tags' and pass to super().__init__."""
        for file_path in self.expectations_dir.rglob("*.py"):
            if file_path.name == "__init__.py":
                continue

            try:
                tree = ast.parse(file_path.read_text())
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ClassDef) or not self._is_expectation_class(node):
                        continue

                    init_method = self._find_init_method(node)
                    if not init_method:
                        continue

                    if not self._has_tags_param(init_method):
                        self.issues.append(
                            f"❌ {node.name} in {file_path.name} missing 'tags' param in __init__"
                        )

                    if not self._has_super_init_with_tags(init_method):
                        self.issues.append(
                            f"❌ {node.name} in {file_path.name} missing super().__init__(tags=tags)"
                        )
            except Exception as e:
                print(f"   ⚠️  Warning: Could not parse {file_path.name}: {e}")

    @staticmethod
    def _is_expectation_class(node: ast.ClassDef) -> bool:
        """Check if a class inherits from DataFrameExpectation (direct or indirect)."""
        return any(
            isinstance(base, ast.Name) and (base.id == "DataFrameExpectation" or base.id.endswith("Expectation"))
            for base in node.bases
        )

    @staticmethod
    def _find_init_method(class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Find the __init__ method in a class."""
        return next(
            (item for item in class_node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"),
            None
        )

    @staticmethod
    def _has_tags_param(init_method: ast.FunctionDef) -> bool:
        """Check if __init__ has 'tags' parameter with correct type annotation."""
        valid_annotations = {
            "Optional[List[str]]", "List[str] | None", "Union[List[str], None]",
            "Optional[list[str]]", "list[str] | None"
        }

        for arg in init_method.args.args:
            if arg.arg == "tags":
                if not arg.annotation:
                    return True
                return ast.unparse(arg.annotation) in valid_annotations
        return False

    @staticmethod
    def _has_super_init_with_tags(init_method: ast.FunctionDef) -> bool:
        """Check if __init__ calls super().__init__(tags=tags)."""
        for stmt in ast.walk(init_method):
            if (isinstance(stmt, ast.Call) and
                isinstance(stmt.func, ast.Attribute) and stmt.func.attr == "__init__" and
                isinstance(stmt.func.value, ast.Call) and
                isinstance(stmt.func.value.func, ast.Name) and stmt.func.value.func.id == "super"):

                if any(kw.arg == "tags" for kw in stmt.keywords):
                    return True
        return False

    def _print_results(self):
        """Print the final results of the sanity check."""
        print("\n" + "=" * 70)
        print("📊 SANITY CHECK RESULTS")
        print("=" * 70)

        print("\n📈 Summary:")
        print(f"   • Registered expectations: {len(self.registered_expectations)}")
        print(f"   • Suite methods:           {len(self.suite_methods)}")
        print(f"   • Test files:              {len(self.test_files)}")
        print(f"   • Issues found:            {len(self.issues)}")

        if self.issues:
            print(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
            print("-" * 40)
            print("\n".join(self.issues))
        else:
            print("\n✅ ALL CHECKS PASSED!")
            print("   The expectations framework is consistent across:")
            print("   • Registry registrations")
            print("   • Suite method implementations")
            print("   • Unit test coverage")

        print("\n" + "=" * 70)

    def print_detailed_mappings(self):
        """Print detailed mappings for debugging purposes."""
        print("\n🔍 DETAILED MAPPINGS")
        print("=" * 50)

        mappings = [
            (f"📋 Registered Expectations ({len(self.registered_expectations)})",
             sorted((f"{name} ({Path(path).name})", name, path) for name, path in self.registered_expectations.items())),
            (f"🎯 Suite Methods ({len(self.suite_methods)})",
             sorted((f"{method}()", method, None) for method in self.suite_methods)),
            (f"🧪 Test Files ({len(self.test_files)})",
             sorted((f"{name} -> {Path(path).name}", name, path) for name, path in self.test_files.items())),
        ]

        for title, items in mappings:
            print(f"\n{title}:")
            for display, *_ in items:
                print(f"   • {display}")


def main():
    """Main entry point for the sanity check script."""
    project_root = Path(__file__).parent.parent

    # Add project root to sys.path for imports
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Validate directory structure
    expected_paths = ["dataframe_expectations", "tests", "pyproject.toml"]
    missing = [p for p in expected_paths if not (project_root / p).exists()]

    if missing:
        print(f"❌ Missing expected directories/files: {missing}")
        print(f"   Script location: {Path(__file__)}")
        print(f"   Project root: {project_root}")
        sys.exit(1)

    checker = ExpectationsSanityChecker(project_root)
    success = checker.run_full_check()

    if "--verbose" in sys.argv or "-v" in sys.argv:
        checker.print_detailed_mappings()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
