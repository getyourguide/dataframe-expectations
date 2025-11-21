"""
DataFrame Expectations Framework Sanity Check Script

This script validates consistency across the entire expectations framework by checking:
1. All expectations implemented in the expectations/ directory are registered in the registry
2. All registered expectations have corresponding expect_* methods in DataFrameExpectationsSuite
3. All registered expectations have corresponding unit tests in tests/dataframe_expectations/expectations_implemented/

Usage:
    python sanity_check_expectations.py
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


class ExpectationsSanityChecker:
    """Validates consistency across the expectations framework."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.expectations_dir = project_root / "dataframe_expectations" / "expectations"
        self.suite_file = project_root / "dataframe_expectations" / "suite.py"
        self.stub_file = project_root / "dataframe_expectations" / "suite.pyi"
        self.tests_dir = project_root / "tests" / "expectations"

        # Results storage
        self.registered_expectations = {}  # expectation_name -> file_path
        self.suite_methods = set()  # expect_* method names
        self.test_files = {}  # expectation_name -> test_file_path

        # Issues tracking
        self.issues = []

    def run_full_check(self) -> bool:
        """Run all consistency checks and return True if all pass."""
        print("🔍 Starting DataFrame Expectations Framework Sanity Check...")
        print("=" * 70)

        # Step 1: Discover registered expectations
        print("\n📋 Step 1: Discovering registered expectations...")
        self._discover_registered_expectations()
        print(f"   Found {len(self.registered_expectations)} registered expectations")

        # Step 2: Discover suite methods
        print("\n🎯 Step 2: Discovering suite methods...")
        self._discover_suite_methods()
        print(f"   Found {len(self.suite_methods)} expect_* methods in suite")

        # Step 3: Discover test files
        print("\n🧪 Step 3: Discovering test files...")
        self._discover_test_files()
        print(f"   Found {len(self.test_files)} test files")

        # Step 4: Validate stub file
        print("\n📝 Step 4: Validating stub file...")
        self._validate_stub_file()

        # Step 5: Validate consistency
        print("\n✅ Step 5: Validating consistency...")
        self._validate_registry_to_suite_mapping()
        self._validate_registry_to_tests_mapping()
        self._validate_orphaned_suite_methods()
        self._validate_orphaned_test_files()

        # Step 6: Check expectation constructors for tags param and super call
        self._check_expectation_constructor_tags()

        # Report results
        self._print_results()

        return len(self.issues) == 0

    def _discover_registered_expectations(self):
        """Find all @register_expectation decorators in expectation files."""
        expectation_files = list(self.expectations_dir.rglob("*.py"))

        for file_path in expectation_files:
            if file_path.name == "__init__.py":
                continue

            try:
                with open(file_path, "r") as f:
                    content = f.read()

                # Parse AST to find @register_expectation decorators
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for decorator in node.decorator_list:
                            if self._is_register_expectation_decorator(decorator):
                                expectation_name = self._extract_expectation_name(decorator)
                                if expectation_name:
                                    self.registered_expectations[expectation_name] = str(file_path)

            except Exception as e:
                print(f"   ⚠️  Warning: Could not parse {file_path}: {e}")

    def _is_register_expectation_decorator(self, decorator) -> bool:
        """Check if a decorator is @register_expectation."""
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name) and decorator.func.id == "register_expectation":
                return True
        return False

    def _extract_expectation_name(self, decorator) -> Optional[str]:
        """Extract expectation name from @register_expectation("Name") decorator."""
        if isinstance(decorator, ast.Call) and decorator.args:
            first_arg = decorator.args[0]
            if isinstance(first_arg, ast.Constant):
                return str(first_arg.value)
        return None

    def _discover_suite_methods(self):
        """Find all expect_* methods available via the registry."""
        try:
            from dataframe_expectations.registry import (
                DataFrameExpectationRegistry,
            )

            # Get the mapping of suite methods from the registry
            mapping = DataFrameExpectationRegistry.get_suite_method_mapping()
            self.suite_methods = set(mapping.keys())

        except Exception as e:
            self.issues.append(f"❌ Could not load suite methods from registry: {e}")

    def _discover_test_files(self):
        """Find all test files and map them to expectation names."""
        if not self.tests_dir.exists():
            self.issues.append(f"❌ Tests directory not found: {self.tests_dir}")
            return

        test_files = list(self.tests_dir.rglob("test_*.py"))

        for test_file in test_files:
            # Skip template files
            if "template" in test_file.name.lower():
                continue

            # Extract potential expectation name from filename
            # e.g., test_expect_value_equals.py -> ExpectationValueEquals
            filename = test_file.stem
            if filename.startswith("test_expect_"):
                # Convert test_expect_value_equals -> ValueEquals
                expectation_part = filename[12:]  # Remove "test_expect_"
                expectation_name = "Expectation" + self._snake_to_pascal_case(expectation_part)
                self.test_files[expectation_name] = str(test_file)

    def _validate_stub_file(self):
        """Check if the stub file is up-to-date with registered expectations."""
        print("   📝 Checking if stub file is up-to-date...")

        if not self.stub_file.exists():
            self.issues.append(
                f"❌ Stub file not found: {self.stub_file}\n"
                "     Run: python scripts/generate_suite_stubs.py"
            )
            return

        try:
            from generate_suite_stubs import generate_pyi_file

            # Generate what the content should be
            expected_content = generate_pyi_file()

            # Read the current stub file
            with open(self.stub_file, "r") as f:
                actual_content = f.read()

            if expected_content != actual_content:
                self.issues.append(
                    "❌ Stub file is out of date\n"
                    "     Run: python scripts/generate_suite_stubs.py"
                )
        except Exception as e:
            self.issues.append(f"❌ Could not validate stub file: {e}")

    def _snake_to_pascal_case(self, snake_str: str) -> str:
        """Convert snake_case to PascalCase."""
        components = snake_str.split("_")
        return "".join(word.capitalize() for word in components)

    def _validate_registry_to_suite_mapping(self):
        """Check that all registered expectations have suite methods."""
        print("   🔗 Checking registry -> suite mapping...")

        missing_suite_methods = []

        for expectation_name in self.registered_expectations.keys():
            # Convert expectation name to expected suite method name
            expected_method = self._expectation_to_suite_method(expectation_name)

            if expected_method not in self.suite_methods:
                missing_suite_methods.append((expectation_name, expected_method))

        if missing_suite_methods:
            self.issues.append("❌ Registered expectations missing suite methods:")
            for exp_name, method_name in missing_suite_methods:
                self.issues.append(f"     • {exp_name} -> missing {method_name}()")

    def _validate_registry_to_tests_mapping(self):
        """Check that all registered expectations have test files."""
        print("   🧪 Checking registry -> tests mapping...")

        missing_tests = []

        for expectation_name in self.registered_expectations.keys():
            if expectation_name not in self.test_files:
                missing_tests.append(expectation_name)

        if missing_tests:
            self.issues.append("❌ Registered expectations missing test files:")
            for exp_name in missing_tests:
                expected_test_file = self._expectation_to_test_filename(exp_name)
                self.issues.append(f"     • {exp_name} -> missing {expected_test_file}")

    def _validate_orphaned_suite_methods(self):
        """Check for suite methods without corresponding registered expectations."""
        print("   🔍 Checking for orphaned suite methods...")

        orphaned_methods = []

        for method_name in self.suite_methods:
            expected_expectation = self._suite_method_to_expectation(method_name)

            if expected_expectation not in self.registered_expectations:
                orphaned_methods.append((method_name, expected_expectation))

        if orphaned_methods:
            self.issues.append("❌ Suite methods without registered expectations:")
            for method_name, exp_name in orphaned_methods:
                self.issues.append(f"     • {method_name}() -> missing {exp_name}")

    def _validate_orphaned_test_files(self):
        """Check for test files without corresponding registered expectations."""
        print("   🧪 Checking for orphaned test files...")

        orphaned_tests = []

        for expectation_name, test_file in self.test_files.items():
            if expectation_name not in self.registered_expectations:
                orphaned_tests.append((expectation_name, test_file))

        if orphaned_tests:
            self.issues.append("❌ Test files without registered expectations:")
            for exp_name, test_file in orphaned_tests:
                self.issues.append(f"     • {test_file} -> missing {exp_name}")

    def _expectation_to_suite_method(self, expectation_name: str) -> str:
        """Convert expectation name to expected suite method name."""
        # Remove "Expectation" prefix if present
        if expectation_name.startswith("Expectation"):
            name_part = expectation_name[11:]  # Remove "Expectation"
        else:
            name_part = expectation_name

        # Convert PascalCase to snake_case and add "expect_" prefix
        snake_case = re.sub("([A-Z])", r"_\1", name_part).lower().lstrip("_")
        return f"expect_{snake_case}"

    def _suite_method_to_expectation(self, method_name: str) -> str:
        """Convert suite method name to expected expectation name."""
        if method_name.startswith("expect_"):
            name_part = method_name[7:]  # Remove "expect_"
            # Convert snake_case to PascalCase and add "Expectation" prefix
            pascal_case = self._snake_to_pascal_case(name_part)
            return f"Expectation{pascal_case}"
        return method_name

    def _expectation_to_test_filename(self, expectation_name: str) -> str:
        """Convert expectation name to expected test filename."""
        method_name = self._expectation_to_suite_method(expectation_name)
        return f"test_{method_name}.py"

    def _check_expectation_constructor_tags(self):
        """Check that all DataFrameExpectation subclasses accept 'tags' in their constructor and pass to super().__init__."""
        print("   🏷️  Checking expectation constructors for 'tags' param and super call...")

        for file_path in self.expectations_dir.rglob("*.py"):
            if file_path.name == "__init__.py":
                continue

            try:
                tree = ast.parse(file_path.read_text())

                for node in ast.walk(tree):
                    if not isinstance(node, ast.ClassDef):
                        continue

                    if not self._is_expectation_class(node):
                        continue

                    init_method = self._find_init_method(node)
                    if not init_method:
                        continue

                    # Check for 'tags' param
                    if not self._has_tags_param(init_method):
                        self.issues.append(
                            f"❌ {node.name} in {file_path.name} missing 'tags' param in __init__"
                        )

                    # Check for super().__init__(tags=tags)
                    if not self._has_super_init_with_tags(init_method):
                        self.issues.append(
                            f"❌ {node.name} in {file_path.name} missing super().__init__(tags=tags) call"
                        )

            except Exception as e:
                print(f"   ⚠️  Warning: Could not parse {file_path}: {e}")

    def _is_expectation_class(self, node: ast.ClassDef) -> bool:
        """Check if a class inherits from DataFrameExpectation (direct or indirect)."""
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id == "DataFrameExpectation" or base.id.endswith("Expectation"):
                    return True
        return False

    def _find_init_method(self, class_node: ast.ClassDef) -> Optional[ast.FunctionDef]:
        """Find the __init__ method in a class."""
        for item in class_node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                return item
        return None

    def _has_tags_param(self, init_method: ast.FunctionDef) -> bool:
        """Check if __init__ method has 'tags' parameter with correct type annotation."""
        for arg in init_method.args.args:
            if arg.arg == "tags":
                # Check if the parameter has the correct type annotation: Optional[List[str]]
                if arg.annotation:
                    annotation_str = ast.unparse(arg.annotation)
                    # Accept various valid forms: Optional[List[str]], List[str] | None, Union[List[str], None]
                    valid_annotations = [
                        "Optional[List[str]]",
                        "List[str] | None",
                        "Union[List[str], None]",
                        "Optional[list[str]]",
                        "list[str] | None"
                    ]
                    if annotation_str not in valid_annotations:
                        return False
                return True
        return False

    def _has_super_init_with_tags(self, init_method: ast.FunctionDef) -> bool:
        """Check if __init__ calls super().__init__(tags=tags)."""
        for stmt in ast.walk(init_method):
            if not isinstance(stmt, ast.Call):
                continue

            # Check if this is a call to __init__
            if not (isinstance(stmt.func, ast.Attribute) and stmt.func.attr == "__init__"):
                continue

            # Check if it's called on super()
            if not (isinstance(stmt.func.value, ast.Call) and
                    isinstance(stmt.func.value.func, ast.Name) and
                    stmt.func.value.func.id == "super"):
                continue

            # Check if tags is passed as keyword argument
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
        print(f"   • Suite methods:          {len(self.suite_methods)}")
        print(f"   • Test files:             {len(self.test_files)}")
        print(f"   • Issues found:           {len(self.issues)}")

        if self.issues:
            print(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
            print("-" * 40)
            for issue in self.issues:
                print(issue)
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

        print(f"\n📋 Registered Expectations ({len(self.registered_expectations)}):")
        for name, file_path in sorted(self.registered_expectations.items()):
            print(f"   • {name} ({Path(file_path).name})")

        print(f"\n🎯 Suite Methods ({len(self.suite_methods)}):")
        for method in sorted(self.suite_methods):
            print(f"   • {method}()")

        print(f"\n🧪 Test Files ({len(self.test_files)}):")
        for name, file_path in sorted(self.test_files.items()):
            print(f"   • {name} -> {Path(file_path).name}")

    def should_run_check(self) -> bool:
        """Check if we should run based on changed files in the current branch."""
        import subprocess

        try:
            # Try to get the default branch name (usually main or master)
            try:
                result = subprocess.run(
                    ["git", "symbolic-ref", "refs/remotes/origin/HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                default_branch = result.stdout.strip().split("/")[-1]
            except subprocess.CalledProcessError:
                # Fallback to common default branch names
                for branch in ["main", "master"]:
                    try:
                        subprocess.run(
                            ["git", "rev-parse", f"origin/{branch}"],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        default_branch = branch
                        break
                    except subprocess.CalledProcessError:
                        continue
                else:
                    default_branch = "main"  # Final fallback

            # Get list of changed files compared to default branch
            result = subprocess.run(
                ["git", "diff", f"origin/{default_branch}...HEAD", "--name-only"],
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = [f for f in result.stdout.strip().split("\n") if f]

            if not changed_files:
                print("🔍 No files changed, skipping sanity check.")
                return False

            # Check if any relevant files changed
            relevant_patterns = [
                "dataframe_expectations/",
                "tests/dataframe_expectations/",
            ]

            changed_relevant_files = []
            for file in changed_files:
                for pattern in relevant_patterns:
                    if pattern in file:
                        changed_relevant_files.append(file)
                        break

            if changed_relevant_files:
                print("🔍 Relevant DataFrame expectations files changed:")
                for file in changed_relevant_files:
                    print(f"   • {file}")
                return True
            else:
                print("🔍 No relevant DataFrame expectations files changed, skipping sanity check.")
                return False

        except subprocess.CalledProcessError as e:
            print(f"⚠️  Git command failed: {e}")
            print("🔍 Running sanity check anyway as a safety measure.")
            return True
        except Exception as e:
            print(f"⚠️  Error checking changed files: {e}")
            print("🔍 Running sanity check anyway as a safety measure.")
            return True


if __name__ == "__main__":
    # Use relative path from the script location
    script_dir = Path(__file__).parent
    # Go up one level: sanity_checks.py is in scripts/, project root is parent
    project_root = script_dir.parent

    # Add project root to sys.path to enable imports
    # This must be done before creating the checker since imports happen during initialization
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Validate directory structure
    expected_dirs = ["dataframe_expectations", "tests", "pyproject.toml"]
    missing_dirs = [d for d in expected_dirs if not (project_root / d).exists()]

    if missing_dirs:
        print(f"❌ Missing expected directories/files: {missing_dirs}")
        print(f"Script location: {Path(__file__)}")
        print(f"Project root: {project_root}")
        sys.exit(1)

    checker = ExpectationsSanityChecker(project_root)

    # Run the checks
    success = checker.run_full_check()

    # Optionally print detailed mappings for debugging
    if "--verbose" in sys.argv or "-v" in sys.argv:
        checker.print_detailed_mappings()

    # Exit with appropriate code
    sys.exit(0 if success else 1)
