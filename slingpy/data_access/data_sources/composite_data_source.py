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
import numpy as np
from itertools import chain
from functools import reduce
from collections import OrderedDict
from typing import List, Any, NoReturn, AnyStr, Dict, Tuple, Union, Optional
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


class CompositeDataSource(AbstractDataSource):
    """
    A composite data source aggregating multiple sub-data sources with items potentially sharing the same row names.
    """
    def __init__(self, data_sources: List[AbstractDataSource], included_indices: Optional[List[int]] = None,
                 flatten: bool = False):
        self.flatten = flatten
        self.data_sources = data_sources
        self._build_row_index()
        if included_indices is None:
            # Include all indices in sub data sources by default.
            all_keys = sorted(set(
                chain.from_iterable([data_source.get_row_names() for data_source in self.data_sources])
            ))
            included_indices = list(chain.from_iterable([self.row_index[k] for k in all_keys]))
        super(CompositeDataSource, self).__init__(included_indices=included_indices, row_index=self.row_index)
        self._check_included_ids_format()

    def get_shape(self) -> List[Tuple[Union[Optional[int]]]]:
        shapes = []
        for data_source in self.data_sources:
            shapes += data_source.get_shape()
        return shapes

    def _get_data(self) -> List[np.ndarray]:
        all_data = [self[idx] for idx in range(len(self))]
        all_data = list(map(lambda x: np.stack(x), zip(*all_data)))
        assert np.alltrue(np.equal(list(map(len, all_data)), len(self))), \
            "The returned data should have __len(self)__ rows."
        return all_data

    def _check_included_ids_format(self) -> NoReturn:
        all_identifiers = set(range(len(self.reverse_row_index)))
        assert set(self.included_indices).issubset(all_identifiers), \
            "All included IDs must be present in the identifier list."

    @staticmethod
    def _increment_state(state, lengths):
        for i in range(len(lengths)):
            state[i] += 1
            if state[i] % lengths[i] == 0:
                state[i] = 0
            else:
                break
        return state

    def _build_row_index(self) -> NoReturn:
        for data_source in self.data_sources:
            if data_source.row_index is None:
                data_source._build_row_index()

        row_indices = list(map(lambda x: x.row_index, self.data_sources))
        keys = sorted(set(chain.from_iterable(map(lambda x: list(x.keys()), row_indices))))
        self.reverse_row_index = OrderedDict(zip(range(len(keys)), keys))

        self.row_index = OrderedDict()
        for key, value in zip(keys, range(len(keys))):
            self.row_index[key] = [value]

        if self.flatten:
            self.offset_index = OrderedDict()
            offset = 0
            for key in keys:
                vals = list(map(lambda x: x.get_by_row_name(key), self.data_sources))
                lengths = list(map(lambda x: max(len(x), 1), vals))
                combinations = reduce(lambda a, b: a*b, lengths)
                state = [0 for _ in range(len(lengths))]

                self.offset_index[self.row_index[key][0]] = state
                state = CompositeDataSource._increment_state(state, lengths)

                for comb_offset in range(1, combinations):
                    cur_index = len(keys) + offset
                    self.reverse_row_index[cur_index] = key
                    self.row_index[key].append(cur_index)
                    self.offset_index[cur_index] = state
                    offset += 1
                    state = CompositeDataSource._increment_state(state, lengths)

    def __len__(self) -> int:
        return len(self.included_indices)

    def __getitem__(self, index: int) -> List[np.ndarray]:
        forwarded_index = self.included_indices[index]
        name_for_index = self.reverse_row_index[forwarded_index]

        if self.flatten:
            state = self.offset_index[forwarded_index]
        else:
            state = [0]*len(self.data_sources)

        values = []
        for this_idx, data_source in zip(state, self.data_sources):
            value = data_source.get_by_row_name(name_for_index)
            if self.flatten:
                value = value[this_idx:this_idx+1]
            values.append(value)
        return list(chain.from_iterable(values))

    def get_by_row_name(self, row_name: AnyStr) -> List[np.ndarray]:
        if self.row_index is None:
            self._build_row_index()
        return list(chain.from_iterable([data_source.get_by_row_name(row_name) for data_source in self.data_sources]))

    def _get_all_row_names(self) -> List[AnyStr]:
        return [self.reverse_row_index[index] for index in range(len(self.reverse_row_index))]

    def get_row_names(self) -> List[AnyStr]:
        if self.row_index is None:
            self._build_row_index()
        return [self.reverse_row_index[index] for index in self.included_indices]

    def get_column_names(self) -> List[List[AnyStr]]:
        column_names = list(map(lambda x: x.get_column_names(), self.data_sources))
        return column_names

    def get_column_code_lists(self) -> List[List[Dict[int, AnyStr]]]:
        columns_code_lists = list(map(lambda x: x.get_column_code_lists(), self.data_sources))
        return columns_code_lists
