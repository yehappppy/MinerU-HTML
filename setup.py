"""Minimal setup.py that reads from pyproject.toml."""

from pathlib import Path

from setuptools import setup

# Read pyproject.toml
pyproject_path = Path(__file__).parent / 'pyproject.toml'

# Parse pyproject.toml to extract metadata
import tomllib

with open(pyproject_path, 'rb') as f:
    config = tomllib.load(f)

project = config['project']

setup(
    name=project['name'],
    version=project['version'],
    description=project['description'],
    readme=project['readme'],
    license=project['license'],
    requires_python=project['requires-python'],
    authors=project['authors'],
    keywords=', '.join(project['keywords']),
    classifiers=project['classifiers'],
    dependencies=project['dependencies'],
    extras_require=project.get('optional-dependencies', {}),
    url=project['urls']['Repository'],
    project_urls=project['urls'],
)
