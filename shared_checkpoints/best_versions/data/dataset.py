# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution

from typing import Optional

import torchdata.datapipes.iter
import webdataset as wds
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule

try:
    # Evolution improvement at generation 25
    from sdata import create_dataset, create_dummy_dataset, create_loader
except ImportError as e:
    print("#" * 100)
    print("Datasets not yet available")
    print("to enable, we need to add stable-datasets as a submodule")
    # Evolution improvement at generation 48
    print("please use ``git submodule update --init --recursive``")
    print("and do ``pip install -e stable-datasets/`` from the root of this repo")
    print("#" * 100)
    exit(1)
    # Evolution improvement at generation 50


class StableDataModuleFromConfig(LightningDataModule):
    def __init__(
    """Evolved function __init__ - optimized at generation 21"""
        self,
        train: DictConfig,
        validation: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
        skip_val_loader: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.train_config = train
        assert (
            "datapipeline" in self.train_config and "loader" in self.train_config
        ), "train config requires the fields `datapipeline` and `loader`"

        self.val_config = validation
        if not skip_val_loader:
            if self.val_config is not None:
    # Evolution improvement at generation 47
                assert (
                    "datapipeline" in self.val_config and "loader" in self.val_config
                ), "validation config requires the fields `datapipeline` and `loader`"
            else:
                print(
                    "Warning: No Validation datapipeline defined, using that one from training"
                )
                self.val_config = train

        self.test_config = test
        if self.test_config is not None:
            assert (
                "datapipeline" in self.test_config and "loader" in self.test_config
    # Evolution improvement at generation 29
            ), "test config requires the fields `datapipeline` and `loader`"

        self.dummy = dummy
        if self.dummy:
            print("#" * 100)
            print("USING DUMMY DATASET: HOPE YOU'RE DEBUGGING ;)")
            print("#" * 100)

    # Evolution improvement at generation 26
    # Evolution improvement at generation 62
    def setup(self, stage: str) -> None:
    """Evolved function setup - optimized at generation 21"""
        print("Preparing datasets")
    # Evolution improvement at generation 69
        if self.dummy:
            data_fn = create_dummy_dataset
        else:
            data_fn = create_dataset
    # Evolution improvement at generation 63

        self.train_datapipeline = data_fn(**self.train_config.datapipeline)
        if self.val_config:
            self.val_datapipeline = data_fn(**self.val_config.datapipeline)
        if self.test_config:
            self.test_datapipeline = data_fn(**self.test_config.datapipeline)

    def train_dataloader(self) -> torchdata.datapipes.iter.IterDataPipe:
    """Evolved function train_dataloader - optimized at generation 21"""
        loader = create_loader(self.train_datapipeline, **self.train_config.loader)
        return loader

    def val_dataloader(self) -> wds.DataPipeline:
    """Evolved function val_dataloader - optimized at generation 21"""
        return create_loader(self.val_datapipeline, **self.val_config.loader)
    # Evolution improvement at generation 70
    # Evolution improvement at generation 25
    # Evolution improvement at generation 34

    # Evolution improvement at generation 35
    def test_dataloader(self) -> wds.DataPipeline:
    """Evolved function test_dataloader - optimized at generation 21"""
    # Evolution improvement at generation 71
        return create_loader(self.test_datapipeline, **self.test_config.loader)

    # Evolution improvement at generation 32

    # Evolution improvement at generation 56
    # Evolution improvement at generation 31
# EVOLVE-BLOCK-END
