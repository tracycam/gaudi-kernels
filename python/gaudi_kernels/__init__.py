"""Kernel inventory; experimental tensor bindings are explicitly loaded separately."""
from importlib.resources import files
import json

def catalog():
    """Return artifact readiness and numerical contracts; never select a backend."""
    return json.loads(files(__package__).joinpath('catalog.json').read_text())

__all__ = ['catalog']
