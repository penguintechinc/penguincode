"""Project language and library detection.

Detects which languages and libraries a project uses by parsing
dependency files. Only detected libraries will have docs indexed,
preventing RAG bloat.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import Language, Library, ProjectContext


# File patterns that indicate a language
LANGUAGE_INDICATORS: Dict[Language, List[str]] = {
    Language.PYTHON: ["*.py", "pyproject.toml", "requirements.txt", "setup.py", "Pipfile"],
    Language.JAVASCRIPT: ["*.js", "*.jsx", "*.mjs", "package.json"],
    Language.TYPESCRIPT: ["*.ts", "*.tsx", "tsconfig.json"],
    Language.GO: ["*.go", "go.mod", "go.sum"],
    Language.RUST: ["*.rs", "Cargo.toml"],
}

# Dependency files and their languages
DEPENDENCY_FILES: Dict[str, Language] = {
    "pyproject.toml": Language.PYTHON,
    "requirements.txt": Language.PYTHON,
    "setup.py": Language.PYTHON,
    "Pipfile": Language.PYTHON,
    "package.json": Language.JAVASCRIPT,  # Also covers TypeScript
    "go.mod": Language.GO,
    "Cargo.toml": Language.RUST,
}


class ProjectDetector:
    """Detects project languages and libraries from dependency files."""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)

    def detect(self) -> ProjectContext:
        """
        Detect languages and libraries in the project.

        Returns:
            ProjectContext with detected languages and libraries
        """
        languages: Set[Language] = set()
        libraries: List[Library] = []
        dependency_files: Dict[str, str] = {}

        # First, detect languages from file extensions
        languages.update(self._detect_languages_from_files())

        # Then parse dependency files for specific libraries
        for filename, language in DEPENDENCY_FILES.items():
            dep_file = self.project_dir / filename
            if dep_file.exists():
                try:
                    content = dep_file.read_text()
                    dependency_files[filename] = content
                    languages.add(language)

                    # Parse libraries from the dependency file
                    libs = self._parse_dependency_file(filename, content, language)
                    libraries.extend(libs)
                except Exception:
                    pass  # Skip files we can't read

        # Check for TypeScript in package.json projects
        if Language.JAVASCRIPT in languages:
            if (self.project_dir / "tsconfig.json").exists():
                languages.add(Language.TYPESCRIPT)

        return ProjectContext(
            languages=list(languages),
            libraries=libraries,
            dependency_files=dependency_files,
        )

    def _detect_languages_from_files(self) -> Set[Language]:
        """Detect languages from file extensions in the project."""
        languages: Set[Language] = set()

        # Only scan top-level and src directories to avoid deep traversal
        scan_dirs = [self.project_dir]
        src_dir = self.project_dir / "src"
        if src_dir.exists():
            scan_dirs.append(src_dir)

        extension_to_language = {
            ".py": Language.PYTHON,
            ".js": Language.JAVASCRIPT,
            ".jsx": Language.JAVASCRIPT,
            ".ts": Language.TYPESCRIPT,
            ".tsx": Language.TYPESCRIPT,
            ".go": Language.GO,
            ".rs": Language.RUST,
        }

        for scan_dir in scan_dirs:
            try:
                for file_path in scan_dir.iterdir():
                    if file_path.is_file():
                        ext = file_path.suffix.lower()
                        if ext in extension_to_language:
                            languages.add(extension_to_language[ext])
            except PermissionError:
                pass

        return languages

    def _parse_dependency_file(
        self, filename: str, content: str, language: Language
    ) -> List[Library]:
        """Parse a dependency file to extract libraries."""
        if filename == "pyproject.toml":
            return self._parse_pyproject_toml(content)
        elif filename == "requirements.txt":
            return self._parse_requirements_txt(content)
        elif filename == "setup.py":
            return self._parse_setup_py(content)
        elif filename == "package.json":
            return self._parse_package_json(content)
        elif filename == "go.mod":
            return self._parse_go_mod(content)
        elif filename == "Cargo.toml":
            return self._parse_cargo_toml(content)
        return []

    def _parse_pyproject_toml(self, content: str) -> List[Library]:
        """Parse pyproject.toml for Python dependencies."""
        libraries = []

        try:
            import tomllib
            data = tomllib.loads(content)

            # Get dependencies from [project.dependencies]
            deps = data.get("project", {}).get("dependencies", [])
            for dep in deps:
                name, version = self._parse_python_requirement(dep)
                if name:
                    libraries.append(Library(
                        name=name,
                        language=Language.PYTHON,
                        version=version,
                    ))

            # Also check [tool.poetry.dependencies]
            poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
            for name, version_spec in poetry_deps.items():
                if name.lower() != "python":
                    version = version_spec if isinstance(version_spec, str) else None
                    libraries.append(Library(
                        name=name,
                        language=Language.PYTHON,
                        version=version,
                    ))

        except Exception:
            # Fallback: regex parsing
            dep_pattern = r'^\s*"([a-zA-Z0-9_-]+)(?:\[.*\])?(?:[<>=!~]+.*)?"\s*,?\s*$'
            for line in content.split('\n'):
                match = re.match(dep_pattern, line)
                if match:
                    libraries.append(Library(
                        name=match.group(1),
                        language=Language.PYTHON,
                    ))

        return libraries

    def _parse_requirements_txt(self, content: str) -> List[Library]:
        """Parse requirements.txt for Python dependencies."""
        libraries = []

        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-'):
                continue

            name, version = self._parse_python_requirement(line)
            if name:
                libraries.append(Library(
                    name=name,
                    language=Language.PYTHON,
                    version=version,
                ))

        return libraries

    def _parse_python_requirement(self, requirement: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse a Python requirement string like 'package>=1.0.0'."""
        # Remove comments and extras
        requirement = requirement.split('#')[0].strip()
        requirement = re.sub(r'\[.*\]', '', requirement)

        # Match package name and optional version
        match = re.match(r'^([a-zA-Z0-9_-]+)(?:([<>=!~]+)(.+))?$', requirement)
        if match:
            name = match.group(1)
            version = match.group(3).strip() if match.group(3) else None
            return name, version
        return None, None

    def _parse_setup_py(self, content: str) -> List[Library]:
        """Parse setup.py for Python dependencies (basic extraction)."""
        libraries = []

        # Look for install_requires list
        install_requires = re.search(
            r'install_requires\s*=\s*\[(.*?)\]',
            content,
            re.DOTALL
        )
        if install_requires:
            deps_str = install_requires.group(1)
            # Extract quoted strings
            deps = re.findall(r'["\']([^"\']+)["\']', deps_str)
            for dep in deps:
                name, version = self._parse_python_requirement(dep)
                if name:
                    libraries.append(Library(
                        name=name,
                        language=Language.PYTHON,
                        version=version,
                    ))

        return libraries

    def _parse_package_json(self, content: str) -> List[Library]:
        """Parse package.json for JavaScript/TypeScript dependencies."""
        libraries = []

        try:
            import json
            data = json.loads(content)

            # Get regular dependencies
            deps = data.get("dependencies", {})
            for name, version in deps.items():
                libraries.append(Library(
                    name=name,
                    language=Language.JAVASCRIPT,
                    version=version.lstrip('^~'),
                ))

            # Get dev dependencies (they're often important for tooling)
            dev_deps = data.get("devDependencies", {})
            for name, version in dev_deps.items():
                # Skip common dev-only tools that don't need docs
                if name.startswith('@types/'):
                    continue
                libraries.append(Library(
                    name=name,
                    language=Language.JAVASCRIPT,
                    version=version.lstrip('^~'),
                ))

        except Exception:
            pass

        return libraries

    def _parse_go_mod(self, content: str) -> List[Library]:
        """Parse go.mod for Go dependencies."""
        libraries = []

        # Match require blocks and single requires
        require_block = re.search(r'require\s*\((.*?)\)', content, re.DOTALL)
        if require_block:
            for line in require_block.group(1).split('\n'):
                match = re.match(r'\s*(\S+)\s+(\S+)', line)
                if match:
                    libraries.append(Library(
                        name=match.group(1),
                        language=Language.GO,
                        version=match.group(2),
                    ))

        # Also match single-line requires
        single_requires = re.findall(r'^require\s+(\S+)\s+(\S+)', content, re.MULTILINE)
        for name, version in single_requires:
            libraries.append(Library(
                name=name,
                language=Language.GO,
                version=version,
            ))

        return libraries

    def _parse_cargo_toml(self, content: str) -> List[Library]:
        """Parse Cargo.toml for Rust dependencies."""
        libraries = []

        try:
            import tomllib
            data = tomllib.loads(content)

            deps = data.get("dependencies", {})
            for name, spec in deps.items():
                if isinstance(spec, str):
                    version = spec
                elif isinstance(spec, dict):
                    version = spec.get("version")
                else:
                    version = None

                libraries.append(Library(
                    name=name,
                    language=Language.RUST,
                    version=version,
                ))

        except Exception:
            # Fallback: regex parsing
            in_deps = False
            for line in content.split('\n'):
                if line.strip() == '[dependencies]':
                    in_deps = True
                    continue
                elif line.strip().startswith('['):
                    in_deps = False
                    continue

                if in_deps:
                    match = re.match(r'^([a-zA-Z0-9_-]+)\s*=\s*"([^"]+)"', line)
                    if match:
                        libraries.append(Library(
                            name=match.group(1),
                            language=Language.RUST,
                            version=match.group(2),
                        ))

        return libraries
