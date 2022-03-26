"""
Copyright (C) 2021  Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import glob
import pickle
import inspect
import numpy as np
from ilock import ILock
from abc import abstractmethod
from dataclasses import dataclass
from slingpy.utils.path_tools import PathTools
from slingpy.data_access.merge_strategies import *
from slingpy.data_access.data_sources.hdf5_tools import HDF5Tools
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
from typing import List, AnyStr, Dict, Tuple, Union, Optional, Set, Generator
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource
from slingpy.apps.run_policies.abstract_run_policy import AbstractRunPolicy, RunResult


@dataclass()
class DatasetLoadResult(object):
    data: Union[np.ndarray, Generator]
    name: AnyStr
    column_names: List[AnyStr]
    row_names: Optional[List[AnyStr]] = None
    column_code_lists: List[Dict[int, AnyStr]] = None
    generated_shape: Optional[Tuple] = None
    num_generator_steps: int = -1
    version: AnyStr = "1"


class AbstractHDF5Dataset(AbstractRunPolicy, AbstractDataSource):
    """
    NOTE: All __init__ arguments are expected to be __hash()__-able for caching.
    """
    def __init__(self, save_directory: AnyStr, in_memory: bool = False,
                 fill_missing: bool = True, fill_missing_value: float = float("nan"),
                 duplicate_merge_strategy: AbstractMergeStrategy = NoMergeStrategy()):
        super(AbstractHDF5Dataset, self).__init__(included_indices=[])
        self.save_directory = save_directory
        self._data_source = None
        self.in_memory = in_memory
        """ Whether or not the database should be kept in memory (instead of accessed via the file API). """
        self.fill_missing = fill_missing
        """ Whether or not to impute values queried via __get_by_row_name__ with a non-existent row name. """
        self.fill_missing_value = fill_missing_value
        """ Value to use to impute missing return vectors with. Only active if __fill_missing__ is set to True. """
        self.duplicate_merge_strategy = duplicate_merge_strategy

    def get_data_source(self) -> HDF5DataSource:
        if self._data_source is None:
            self._data_source = self.load()
        return self._data_source

    def _run(self, **kwargs) -> RunResult:
        self.get_data_source()
        return RunResult({}, {})

    @abstractmethod
    def _load(self) -> DatasetLoadResult:
        """
        Loads a dataset.

        Returns:
            A __DatasetLoadResult__ object containing the data array, dataset name, column names and row names.
        """
        raise NotImplementedError()

    def get_ignored_param_names(self) -> Set[AnyStr]:
        init = getattr(HDF5DataSource.__init__, 'deprecated_original', HDF5DataSource.__init__)
        init_signature = inspect.signature(init)
        param_names = set([p.name for p in init_signature.parameters.values()
                           if p.name != 'self' and p.kind != p.VAR_KEYWORD])
        param_names -= {'hdf5_file', 'included_indices'}
        param_names = param_names.union({"save_directory"})
        return param_names

    def remove_ignored_params(self, params: Dict) -> Dict:
        for ignored_param in self.get_ignored_param_names():
            if ignored_param in params:
                del params[ignored_param]  # Remove ignored parameters.
        return params

    def get_h5_file_name(self, folder_path) -> AnyStr:
        """
        Gets a matching cached h5 file with arguments equal to __self.get_params()__ from __folder_path__ if it exists.

        Args:
            folder_path: The folder to search for a matching h5 file. Each h5 file is expected to be paired with a
                          .pickle file that contains the corresponding pickled __AbstractHDF5Dataset__ arguments.

        Returns:
            Returns a matching cached h5 file from __folder_path__ if it exists. Otherwise, returns a new h5 file name.

        """
        search_pattern = os.path.join(folder_path, f"{type(self).__name__}_*.h5")
        matched_file_path, highest_index = None, -1
        current_params = dict(self.get_params())
        current_params = self.remove_ignored_params(current_params)
        for file_path in glob.glob(search_pattern):
            root, _ = os.path.splitext(file_path)
            current_index = int(root.split("_")[-1])
            if highest_index < current_index:
                highest_index = current_index

            config_file_path = f"{root}.pickle"
            if not os.path.exists(config_file_path):
                continue  # Skip this HDF5 file since we can not confirm it was created with matching arguments.

            with open(config_file_path, "rb") as fp:
                config = pickle.load(fp)
            config = self.remove_ignored_params(config)

            if AbstractDataSource.safe_dict_compare(config, current_params):
                matched_file_path = file_path
                break
        if matched_file_path is None:
            h5_file_name = os.path.join(folder_path, f"{type(self).__name__}_{highest_index+1:d}.h5")
            return h5_file_name
        else:
            return matched_file_path

    def load(self) -> HDF5DataSource:
        # Search by argument hash.
        ignored_params = self.get_ignored_param_names()
        folder_path = os.path.join(self.save_directory, f"{type(self).__name__}_"
                                                        f"{self.get_params_hash(ignored_params)}")
        with ILock(folder_path, lock_directory=self.save_directory):
            if not os.path.exists(folder_path):
                PathTools.mkdir_if_not_exists(folder_path)

            h5_file_path = self.get_h5_file_name(folder_path)  # Resolve hash conflicts.
            if not os.path.exists(h5_file_path):
                root, _ = os.path.splitext(h5_file_path)
                config_file_path = f"{root}.pickle"
                load_result = self._load()

                if isinstance(load_result.data, Generator):
                    HDF5Tools.save_h5_file_streamed(h5_file_path,
                                                    generator=load_result.data,
                                                    generator_steps=load_result.num_generator_steps,
                                                    generated_shape=load_result.generated_shape,
                                                    dataset_name=load_result.name,
                                                    dataset_version=load_result.version,
                                                    column_names=load_result.column_names,
                                                    column_code_lists=load_result.column_code_lists)
                else:
                    HDF5Tools.save_h5_file(h5_file_path,
                                           load_result.data,
                                           load_result.name,
                                           column_names=load_result.column_names,
                                           row_names=load_result.row_names,
                                           column_code_lists=load_result.column_code_lists,
                                           dataset_version=load_result.version)

                self._save_config(config_file_path)
        return HDF5DataSource(h5_file_path,
                              in_memory=self.in_memory,
                              fill_missing=self.fill_missing,
                              fill_missing_value=self.fill_missing_value,
                              duplicate_merge_strategy=self.duplicate_merge_strategy)

    def _save_config(self, config_file_path):
        params = self.get_params()
        for k, v in params.items():
            if isinstance(v, AbstractHDF5Dataset) or isinstance(v, HDF5DataSource):
                v.reset_cache()  # Cached file handles cannot be pickled.
        with open(config_file_path, "wb") as fp:
            pickle.dump(params, fp)  # Store a config file alongside h5 file.

    def __hash__(self):
        return self.get_data_source().__hash__()

    def __len__(self) -> int:
        return self.get_data_source().__len__()

    def __getitem__(self, index: int) -> List[np.ndarray]:
        return self.get_data_source().__getitem__(index)

    def reset_cache(self):
        self.get_data_source().reset_cache()

    @property
    def file_path(self):
        return self.get_data_source().file_path

    def get_shape(self) -> List[Tuple[Union[Optional[int]]]]:
        return self.get_data_source().get_shape()

    def _get_data(self) -> List[np.ndarray]:
        return self.get_data_source()._get_data()

    def get_by_row_name(self, row_name: AnyStr) -> List[np.ndarray]:
        return self.get_data_source().get_by_row_name(row_name)

    def get_row_names(self) -> List[AnyStr]:
        return self.get_data_source().get_row_names()

    def _get_all_row_names(self) -> List[AnyStr]:
        return self.get_data_source()._get_all_row_names()

    def get_column_names(self) -> List[List[AnyStr]]:
        return self.get_data_source().get_column_names()

    def get_column_code_lists(self) -> List[List[Dict[int, AnyStr]]]:
        return self.get_data_source().get_column_code_lists()

    def subset(self, names: List[AnyStr]) -> "AbstractDataSource":
        return AbstractDataSource.subset(self.get_data_source(), names)

    def _build_row_index(self):
        AbstractDataSource._build_row_index(self.get_data_source())

    def is_variable_length(self) -> bool:
        return AbstractDataSource.is_variable_length(self.get_data_source())

    def to_torch(self) -> "TorchDataSource":
        return AbstractDataSource.to_torch(self.get_data_source())
