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
import h5py
import hashlib
import numpy as np
from slingpy.data_access.merge_strategies import *
from typing import Union, List, IO, AnyStr, NoReturn, Dict, Tuple, Optional
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


class HDF5DataSource(AbstractDataSource):
    """
    A data source using the h5py backend for data access.
    """
    COVARIATES_KEY = "covariates"
    SHAPES_KEY = "shapes"
    ROWNAMES_KEY = "rownames"
    COLNAMES_KEY = "colnames"
    COLUMN_CODE_LIST_KEYS_KEY = "colmapkeys_col{:d}"
    COLUMN_CODE_LIST_VALUES_KEY = "colmapvalues_col{:d}"
    DATASET_NAME = "name"
    DATASET_VERSION = "version"

    def __init__(self, hdf5_file: Union[AnyStr, IO], included_indices: Optional[List[int]] = None,
                 in_memory: bool = False, fill_missing: bool = True, fill_missing_value: float = float("nan"),
                 duplicate_merge_strategy: AbstractMergeStrategy = NoMergeStrategy()):
        super(HDF5DataSource, self).__init__(included_indices)

        self.hdf5_file = hdf5_file
        """ Path to the hd5 file that is used for the underlying data storage. """
        self.file_cache = None
        self.row_index = None
        self.in_memory = in_memory
        """ Whether or not the database should be kept in memory (instead of accessed via the file API). """
        self.fill_missing = fill_missing
        """ Whether or not to impute values queried via __get_by_row_name__ with a non-existent row name. """
        self.fill_missing_value = fill_missing_value
        """ Value to use to impute missing return vectors with. Only active if __fill_missing__ is set to True. """
        self.duplicate_merge_strategy = duplicate_merge_strategy
        """ 
        Strategy to use to resolve cases where multiple entries exist for a single row name. 
        By default no resolution is applied and all values are returned. 
        """

        if included_indices is None:
            # Include all indices by default.
            self.included_indices = np.array(list(range(self._underlying_length())))
        else:
            self.included_indices = np.array(included_indices)

        self._check_hd5_file_format()
        self._check_included_ids_format()

        if self.in_memory:
            self.data = self.get_data()
        else:
            self.data = None
        self.needs_reshape = HDF5DataSource.SHAPES_KEY in self._get_file_cache()

    @property
    def file_path(self):
        key = self.hdf5_file
        if hasattr(self.hdf5_file, "name"):
            key = self.hdf5_file.name
        return key

    def __hash__(self):
        key = self.file_path
        m = hashlib.sha256()
        m.update(bytes(key, 'utf-8'))
        return int(m.hexdigest(), 16)

    def _get_file_cache(self):
        if self.file_cache is None:
            self.file_cache = h5py.File(self.hdf5_file, "r")
        return self.file_cache

    def reset_cache(self):
        self.file_cache.close()
        self.file_cache = None

    def get_shape(self) -> List[Tuple[Union[Optional[int]]]]:
        hd5_file = self._get_file_cache()
        shape = hd5_file[HDF5DataSource.COVARIATES_KEY].shape

        has_dynamic_dimensions = len(shape) == 1
        if has_dynamic_dimensions:
            if HDF5DataSource.SHAPES_KEY in hd5_file:
                original_shape = (None,)*len(hd5_file[HDF5DataSource.SHAPES_KEY].shape)
            else:
                original_shape = (None,)
            shape = shape + original_shape
            if shape[-1] is None:
                shape = shape + (1,)
        return [shape]

    def _check_hd5_file_format(self) -> NoReturn:
        hd5_file = self._get_file_cache()
        assert HDF5DataSource.COVARIATES_KEY in hd5_file
        assert HDF5DataSource.ROWNAMES_KEY in hd5_file
        assert HDF5DataSource.COLNAMES_KEY in hd5_file
        if len(hd5_file[HDF5DataSource.COVARIATES_KEY].shape) == 2:
            assert len(hd5_file[HDF5DataSource.COLNAMES_KEY]) == hd5_file[HDF5DataSource.COVARIATES_KEY].shape[-1]
        elif len(hd5_file[HDF5DataSource.COVARIATES_KEY].shape) == 1:
            assert len(hd5_file[HDF5DataSource.COLNAMES_KEY]) == 1
        assert len(hd5_file[HDF5DataSource.ROWNAMES_KEY]) == hd5_file[HDF5DataSource.COVARIATES_KEY].shape[0]
        assert HDF5DataSource.DATASET_NAME in hd5_file.attrs
        assert HDF5DataSource.DATASET_VERSION in hd5_file.attrs

    def _check_included_ids_format(self) -> NoReturn:
        all_identifiers = set(range(self._underlying_length()))
        assert set(self.included_indices).issubset(all_identifiers), \
            "All included IDs must be present in the identifier list."

    def _underlying_length(self):
        return self.get_shape()[0][0]

    def _get_at_index(self, index: int) -> np.ndarray:
        hd5_file = self._get_file_cache()
        if self.in_memory:
            data = self.data[0]
        else:
            data = hd5_file[HDF5DataSource.COVARIATES_KEY]
        x = data[index]
        if self.needs_reshape:
            original_shape = hd5_file[HDF5DataSource.SHAPES_KEY][index]
            x = x.reshape(original_shape.astype(np.int32))
        return x

    def __len__(self) -> int:
        return len(self.included_indices)

    def __getitem__(self, index: int) -> List[np.ndarray]:
        forwarded_index = self.included_indices[index]
        return [self._get_at_index(forwarded_index)]

    def get_name(self) -> AnyStr:
        hd5_file = self._get_file_cache()
        return hd5_file.attrs[HDF5DataSource.DATASET_NAME]

    def get_version(self) -> AnyStr:
        hd5_file = self._get_file_cache()
        return hd5_file.attrs[HDF5DataSource.DATASET_VERSION]

    def _get_data(self) -> List[np.ndarray]:
        hd5_file = self._get_file_cache()
        x = hd5_file[HDF5DataSource.COVARIATES_KEY]
        included_indices = np.array([self.included_indices[index] for index in range(len(self))])

        # Query indices must be sorted in order for h5py, re-sort in case __included_indices__ are not sorted.
        ret_val = self._query_h5_sorted(x, included_indices)
        return [ret_val]

    def get_path(self):
        return self.hdf5_file

    def get_column_code_lists(self) -> List[Dict[int, AnyStr]]:
        column_code_lists = []
        hd5_file = self._get_file_cache()
        column_names = self.get_column_names()
        for idx, column in enumerate(column_names):
            keys_name = HDF5DataSource.COLUMN_CODE_LIST_KEYS_KEY.format(idx)
            values_name = HDF5DataSource.COLUMN_CODE_LIST_VALUES_KEY.format(idx)
            if keys_name in hd5_file and values_name in hd5_file:
                keys, values = hd5_file[keys_name], hd5_file[values_name].asstr()[()].tolist()
                if len(keys) != len(values):
                    raise AssertionError(
                        "Malformed hd5 data source. Value map keys and values must be of the same length."
                    )
                column_code_list = dict(zip(keys, values))
            else:
                column_code_list = None
            column_code_lists.append(column_code_list)
        return column_code_lists

    def inverse_transform(self, item):
        column_value_maps = self.get_column_code_lists()
        if len(item) != len(column_value_maps):
            raise AssertionError(f"Item must be the same length as the number of columns in the data source."
                                 f"Expected {len(column_value_maps):d} but was {len(item):d}.")

        item = list(map(lambda value, value_map:
                        value_map[value] if value_map is not None else value,
                        item, column_value_maps))
        return item

    def is_variable_length(self) -> bool:
        if self._underlying_length() == 0:
            return False
        if self.row_index is None:
            self._build_row_index()
        key = list(self.row_index.items())[0][0]
        is_variable_length = isinstance(self.row_index[key], list)
        return is_variable_length

    def get_column_names(self) -> List[AnyStr]:
        hd5_file = self._get_file_cache()
        column_names = hd5_file[HDF5DataSource.COLNAMES_KEY].asstr()
        return column_names[()].tolist()

    def get_column_name(self, index) -> AnyStr:
        hd5_file = self._get_file_cache()
        column_name = hd5_file[HDF5DataSource.COLNAMES_KEY].asstr()[index]
        return column_name

    def _query_h5_sorted(self, h5_array, indices: np.ndarray) -> np.ndarray:
        argsorted_indices = np.argsort(indices)
        reverse_argsort = np.zeros((len(indices),), dtype=int)
        reverse_argsort[argsorted_indices] = np.arange(len(indices), dtype=np.int32)
        sorted_ret_val = h5_array[indices[argsorted_indices]]
        ret_val = sorted_ret_val[reverse_argsort]
        return ret_val

    def _get_all_row_names(self) -> List[AnyStr]:
        hd5_file = self._get_file_cache()
        row_names = hd5_file[HDF5DataSource.ROWNAMES_KEY].asstr()
        return row_names[()]

    def get_row_names(self) -> List[AnyStr]:
        hd5_file = self._get_file_cache()
        row_names = hd5_file[HDF5DataSource.ROWNAMES_KEY].asstr()
        row_names = self._query_h5_sorted(row_names, self.included_indices).tolist()
        return row_names

    def get_row_name(self, index) -> AnyStr:
        forwarded_index = self.included_indices[index]
        hd5_file = self._get_file_cache()
        row_name = hd5_file[HDF5DataSource.ROWNAMES_KEY].asstr()[forwarded_index]
        return row_name

    @staticmethod
    def replace_none_with(shape, value=1):
        return tuple(shape[i] if shape[i] is not None else value for i in range(len(shape)))

    def get_by_row_name(self, row_name: AnyStr) -> List[np.ndarray]:
        if self.row_index is None:
            self._build_row_index()

        if row_name in self.row_index:
            idx = self.row_index[row_name]
        else:
            if self.fill_missing:
                return_shape = self.get_shape()[0][1:]
                return_value = [np.full(HDF5DataSource.replace_none_with(return_shape),
                                        fill_value=self.fill_missing_value)]
                return return_value
            idx = []
        if isinstance(idx, list):
            ret_val = [self._get_at_index(i) for i in idx]
            new_ret_val = self.duplicate_merge_strategy.resolve(ret_val)
            return new_ret_val
        else:
            return [self._get_at_index(idx)]
