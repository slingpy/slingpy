"""
The `slingpy` python package provides starter code and various standard utilities for reproducible machine learning
projects. `slingpy` aims to be a transparent and extensible platform that is agnostic of the modelling-backend
(e.g., Pytorch or Tensorflow) and suitable for both production and research machine learning projects.

.. include:: ./documentation.md
"""
import slingpy.evaluation.metrics as metrics
from slingpy.apps import AbstractBaseApplication
from slingpy.models.torch_model import TorchModel
from slingpy.utils.auto_argparse import AutoArgparse
from slingpy.models.sklearn_model import SklearnModel
from slingpy.utils.to_categorical import to_categorical
from slingpy.utils.argument_dictionary import ArgumentDictionary
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.data_access.stratified_split import StratifiedSplit
from slingpy.data_access.data_sources.hdf5_tools import HDF5Tools
from slingpy.evaluation.metrics.abstract_metric import AbstractMetric
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
from slingpy.data_access.merge_strategies import *
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource
from slingpy.data_access.data_sources.composite_data_source import CompositeDataSource
from slingpy.losses import *
from slingpy.datasets import *
from slingpy.transforms import *
from slingpy.utils.download_streamed import download_streamed
from slingpy.datasets.abstract_hdf5_dataset import AbstractHDF5Dataset
from slingpy.datasets.abstract_hdf5_dataset import DatasetLoadResult


def instantiate_from_command_line(clazz):
    return AutoArgparse.instantiate_from_command_line(clazz)
