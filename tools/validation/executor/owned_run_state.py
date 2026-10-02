"""Persist owned-run lifecycle evidence without truncating the last valid state."""
import json
import os
from pathlib import Path


def save_state(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as file:
        json.dump(value, file, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
