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
import six
import numpy as np
from itertools import chain
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Union, Optional, AnyStr, Dict
from slingpy.utils.argument_dictionary import ArgumentDictionary
from slingpy.data_access.data_sources.torch_data_source import TorchDataSource


@six.add_metaclass(ABCMeta)
class AbstractDataSource(ArgumentDictionary):
    """
    An abstract base data source.
    """
    def __init__(self, included_indices: List[int], row_index: Optional = None):
        self.included_indices = included_indices
        self.row_index = row_index

    @staticmethod
    def safe_dict_compare(d1: Dict, d2: Dict) -> bool:
        for key in d1.keys():
            if key not in d2:
                return False
            elif d1[key] != d2[key] and not ((isinstance(d1[key], float) and np.isnan(d1[key])) and
                                             (isinstance(d2[key], float) and np.isnan(d2[key]))):
                # Consider (NaN == NaN) = True for dict comparison.
                return False
        return True

    def __eq__(self, other: "AbstractDataSource") -> bool:
        if type(other) is type(self):
            return AbstractDataSource.safe_dict_compare(self.get_params(), other.get_params())
        return False

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of samples in this dataset.

        Returns:
            The number of samples in this dataset.
        """
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, index: int) -> List[np.ndarray]:
        """
        Query the item at __index__ from this dataset.

        Args:
            index: The index to query [0, num_samples]

        Returns:
            The item at __index__ in this dataset.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_shape(self) -> List[Tuple[Union[Optional[int]]]]:
        """
        Returns:
           The shape of this dataset with each list entry corresponding to one (sub-)dataset and each list entry
           containing the sizes in each dimension of the dataset. If the dataset is not a composite, the list of dataset
           shapes will be of length 1.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_data(self) -> List[np.ndarray]:
        """
        Returns:
            The entire dataset loaded into memory as a list of __np.ndarray__s. Note that this operation may require a
            large amount of memory. Where possible, this operation should therefore be avoided and entries should be
            accessed individually by index or name instead.
        """
        raise NotImplementedError()

    def get_data(self, row_names: Optional[List[AnyStr]] = None) -> List[np.ndarray]:
        """
        Args:
            row_names: The row names of the items to be retrieved (Default: None, i.e. all data is returned).

        Returns:
            The queried dataset loaded into memory as a list of __np.ndarray__s. Note that this operation may require a
            large amount of memory. Where possible, this operation should therefore be avoided and entries should be
            accessed individually by index or name instead.
        """
        if row_names:
            ret_val = list(map(lambda x: np.stack(x), zip(*[self.get_by_row_name(row_name)
                                                            for row_name in row_names])))
            return ret_val
        else:
            return self._get_data()

    @abstractmethod
    def get_by_row_name(self, row_name: AnyStr) -> List[np.ndarray]:
        """
        Get an item by its row name.

        Args:
            row_name: The row name of the item to be retrieved.

        Returns:
            The item with __row_name__ as its row name.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_row_names(self) -> List[AnyStr]:
        """
        Get the list of row names.

        Returns:
            A list of row names with each list entry corresponding to the name of the item at that index.
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_all_row_names(self) -> List[AnyStr]:
        """
        Get the list of all row names *not* considering __included_indices__.

        Returns:
            A list of row names with each list entry corresponding to the name of the item at that index in the raw
            underlying data.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_column_names(self) -> List[List[AnyStr]]:
        """
        Get the list of column names.

        Returns:
            A list of lists of column names with each list entry corresponding to the name of the column at that index
            for one (sub-)dataset.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_column_code_lists(self) -> List[List[Dict[int, AnyStr]]]:
        """
        Get the list of column code lists, i.e. the mapping of nominal data entries to their names if a column contains
        nominal data points.

        Returns:
            A list of lists of column code lists with each list entry corresponding to the code list associated with
            the column at that index for one (sub-)dataset.
        """
        raise NotImplementedError()

    def subset(self, names: List[AnyStr]) -> "AbstractDataSource":
        if self.row_index is None:
            self._build_row_index()
        indices = list(chain.from_iterable([self.row_index[name] for name in names if name in self.row_index]))
        arguments = self.get_params()
        arguments["included_indices"] = indices
        return self.__class__(**arguments)  # Dynamically create a new class of the same type with the same config.

    def _build_row_index(self):
        row_names = self._get_all_row_names()
        row_index = defaultdict(list)
        for name, index in zip(row_names, range(len(row_names))):
            row_index[name].append(index)
        self.row_index = row_index

    def is_variable_length(self) -> bool:
        """
        Returns:
            Whether or not the dataset is of variable length, i.e. not all entries have the same shape in all
            dimensions.
        """
        return np.any([None in shape for shape in self.get_shape()])

    def to_torch(self) -> TorchDataSource:
        """
        Converts the dataset into a TorchDataSource suitable for use with ´torch´.

        Returns:
            This dataset as a data source compatible with the ´torch´ API.
        """
        return TorchDataSource(self)
