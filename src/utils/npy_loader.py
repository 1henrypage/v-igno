"""Simple helper to load .npy files like h5py."""
import numpy as np
from pathlib import Path


class NpyFile:
    """Load .npy directory like h5py.File()."""

    def __init__(self, path, mode='r'):
        # Change smh_train.mat → smh_train/
        self.path = Path(path)
        if self.path.suffix == '.mat':
            self.path = self.path.parent / self.path.stem

        # Load all .npy files
        self._data = {}
        for f in self.path.glob("*.npy"):
            self._data[f.stem] = np.load(f)

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass