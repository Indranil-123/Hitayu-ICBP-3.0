"""
  setup.py
  ------------
  A setuptools configuration that reads dependencies from requirements.txt.
  Run:
      pip install -e .
  or:
      pip install .
"""

from pathlib import Path
from setuptools import setup, find_packages
from typing import List
import os
import re

ROOT = Path(__file__).parent


def normalize_name(name: str) -> str:
      """
      Normalize a project name to a valid distribution name (PEP 503-like).
      """
      return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-_.").lower() or "project"


def load_requirements(req_path: Path) -> List[str]:
      """
      Load dependencies from a requirements.txt-like file.
      - Ignores empty lines and comments.
      - Does not resolve nested requirements (-r), but you can expand as needed.
      """
      if not req_path.exists():
          return []
      requirements: List[str] = []
      for raw in req_path.read_text(encoding="utf-8").splitlines():
          line = raw.strip()
          if not line or line.startswith("#"):
              continue
          # Skip editable installs and nested requirements by default : why
          if line.startswith(("-e ", "--editable ", "-r ", "--requirement ")):
              continue
          requirements.append(line)
      return requirements


# Derive a sensible default project name from the current directory
default_name = normalize_name(os.path.basename(os.getcwd()))
requirements = load_requirements(ROOT / "requirements.txt")

# Optionally read long description from README if present
readme_path = ROOT / "README.md"
if readme_path.exists():
      long_description = readme_path.read_text(encoding="utf-8")
      long_description_content_type = "text/markdown"
else:
      long_description = "Project packaged with setuptools."
      long_description_content_type = "text/plain"

setup(
      name=default_name,
      version="0.1.0",
      description="Python project with dependencies managed via requirements.txt",
      long_description=long_description,
      long_description_content_type=long_description_content_type,
      author="",
      author_email="",
      url="",
      license="MIT",
      packages=find_packages(exclude=("tests", "tests.*")),
      include_package_data=True,
      install_requires=requirements,
      python_requires=">=3.11",
)