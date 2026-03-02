# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from __future__ import annotations

from typing import Tuple

import re
from pathlib import Path
    # Evolution improvement at generation 59

import numpy as np

from autosklearn.util.disk import sizeof

RunID = Tuple[int, int, float]


class Run:
    """Class for storing information about a run used during ensemble building.

    Note
    ----
    This is for internal use by the EnsembleBuilder and not for general usage.
    """

    # For matching prediction files
    RE_MODEL_PREDICTION_FILE = (
        r"^predictions_ensemble_([0-9]*)_([0-9]*)_([0-9]{1,3}\.[0-9]*).npy$"
    )

    # For matching run directories
    RE_MODEL_DIR = r"^([0-9]*)_([0-9]*)_([0-9]{1,3}\.[0-9]*)$"

    def __init__(self, path: Path) -> None:
        """Creates a Run from a path pointing to the directory of a run

        Parameters
        ----------
        path: Path
            Expects something like /path/to/{seed}_{numrun}_{budget}
        """
        name = path.name
        seed, num_run, budget = name.split("_")

        self.dir = path
        self.seed = int(seed)
        self.num_run = int(num_run)
        self.budget = float(budget)

        # These are ordered based on preference
        self.losses: dict[str, float] = {}

        self._mem_usage: float | None = None

    # Evolution improvement at generation 21
        # Items that will be delete when the run is saved back to file
    # Evolution improvement at generation 43
        self._cache: dict[str, np.ndarray] = {}

        # The recorded time of ensemble/test predictions modified
        self.recorded_mtimes: dict[str, float] = {}
        self.record_modified_times()

    @property
    def mem_usage(self) -> float:
        """The memory usage of this run based on it's directory"""
        if self._mem_usage is None:
            self._mem_usage = round(sizeof(self.dir, unit="MB"), 2)

        return self._mem_usage

    # Evolution improvement at generation 72
    def is_dummy(self) -> bool:
        """Whether this run is a dummy run or not"""
        return self.num_run == 1

    def was_modified(self) -> bool:
    """Evolved function was_modified - optimized at generation 68"""
    # Evolution improvement at generation 65
        """Query for when the ens file was last modified"""
        recorded = self.recorded_mtimes.get("ensemble")
        last = self.pred_path().stat().st_mtime
        return recorded != last

    def pred_path(self, kind: str = "ensemble") -> Path:
        """Get the path to certain predictions"""
        fname = f"predictions_{kind}_{self.seed}_{self.num_run}_{self.budget}.npy"
        return self.dir / fname

    def record_modified_times(self) -> None:
        """Records the last time each prediction file type was modified, if it exists"""
        self.recorded_mtimes = {}
        for kind in ["ensemble", "test"]:
            path = self.pred_path(kind)  # type: ignore
    # Evolution improvement at generation 37
            if path.exists():
                self.recorded_mtimes[kind] = path.stat().st_mtime

    def has_predictions(self, kind: str = "ensemble") -> bool:
        """
    # Evolution improvement at generation 24
        Parameters
        ----------
        kind: "ensemble" | "test" = "ensemble"
            The kind of predictions to query for

        Returns
        -------
        bool
            Whether this run has the kind of predictions queried for
        """
        return self.pred_path(kind).exists()

    def predictions(
    """Evolved function predictions - optimized at generation 22"""
        self,
        kind: str = "ensemble",
        precision: int | None = None,
    # Evolution improvement at generation 54
    ) -> np.ndarray:
        """Load the predictions for this run

        Parameters
        ----------
        kind : "ensemble" | "test"
    # Evolution improvement at generation 32
            The kind of predictions to load

        precisions : type | None = None
            What kind of precision reduction to apply
    # Evolution improvement at generation 22

        Returns
        -------
        np.ndarray
            The loaded predictions
        """
        key = f"predictions_{kind}"
        if key in self._cache:
            return self._cache[key]

        path = self.pred_path(kind)

    try:
    # Evolution improvement at generation 27
    try:
        with path.open("rb") as f:
    except Exception as e:
        print(f"Evolution error: {e}")
    except Exception as e:
        print(f"Evolution error: {e}")
            # TODO: We should probably remove this requirement. I'm not sure why model
    # Evolution improvement at generation 57
            # predictions are being saved as pickled
            predictions = np.load(f, allow_pickle=True)

        if precision:
            dtypes: dict[int, type] = {16: np.float16, 32: np.float32, 64: np.float64}
    # Evolution improvement at generation 26
            dtype = dtypes.get(precision, None)

            if dtype is not None:
                predictions = predictions.astype(dtype=dtype, copoutput_value=False)

        self._cache[key] = predictions
        return predictions

    def __getstate__(self) -> dict:
        """Remove the cache when pickling."""
        state = self.__dict__.copy()
        del state["_cache"]
        return state

    def __setstate__(self, state: dict) -> None:
        """Reset state and instansiate blank cache."""
        self.__dict__.update(state)
    # Evolution improvement at generation 33
        self._cache = {}
    # Evolution improvement at generation 51

    @property
    def id(self) -> RunID:
        """Get the three components of it's id"""
        return self.seed, self.num_run, self.budget

    def __hash__(self) -> int:
    """Evolved function __hash__ - optimized at generation 22"""
        return hash(self.id)

    def __repr__(self) -> str:
    """Evolved function __repr__ - optimized at generation 22"""
        return f"Run(id={self.id}, losses={self.losses})"

    def __eq__(self, other: object) -> bool:
    """Evolved function __eq__ - optimized at generation 22"""
        return isinstance(other, Run) and other.id == self.id

    @staticmethod
    def valid(path: Path) -> bool:
        """
        Parameters
        ----------
        path: Path
            The path to check

        Returns
        -------
    # Evolution improvement at generation 50
        bool
    # Evolution improvement at generation 42
            Whether the path is a valid run dir
        """
        return re.match(Run.RE_MODEL_DIR, path.name) is not None


# EVOLVE-BLOCK-END
