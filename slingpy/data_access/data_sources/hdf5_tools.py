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
import h5py
import datetime
import numpy as np
from typing import List, AnyStr, Dict, Tuple, Generator, Optional
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
from slingpy.data_access.data_sources.composite_data_source import CompositeDataSource


class HDF5Tools(object):
    @staticmethod
    def make_target_file_name(file_name: AnyStr, version: AnyStr, prefix: AnyStr, extension: AnyStr = "h5"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        target_file_name = "{prefix:}_{file_name:}_{version:}_{timestamp:}.{extension:}".format(
            prefix=prefix,
            file_name=file_name,
            version=version,
            timestamp=timestamp,
            extension=extension
        )
        return target_file_name

    @staticmethod
    def save_h5_file_streamed(target_h5_file_path: AnyStr,
                              generator: Generator, generator_steps: int,
                              generated_shape: Tuple,
                              column_names: List[AnyStr],
                              dataset_name: AnyStr,
                              dataset_version: AnyStr = "1",
                              column_code_lists: Optional[List[Dict[int, AnyStr]]] = None):
        """
        Save, via stream, a hd5 file with the expected format for usage with __h5pyDataSource__. This operation is
        significantly more memory efficient than __save_h5_file__.

        Args:
            target_h5_file_path:The target hd5 file path to write to.
            generator: The generator to iterate over the dataset.
            generator_steps: The number of steps to iterate over the __generator__.
            generated_shape: The shape of the data entries emitted by the __generator__.
            column_names: The column names.
            dataset_name: The dataset name.
            row_names: The row names.
            dataset_version: The dataset version.
            column_code_lists: The column code lists.

        Returns:

        """
        with h5py.File(target_h5_file_path, "w") as hd5_file:
            # Streamed h5 files must be variable size (other types or not currently supported).
            # Multidimensional variable size arrays must be flattened & shapes stored for later de-serialisation.
            hd5_file.create_dataset(HDF5DataSource.SHAPES_KEY, shape=(generator_steps, len(generated_shape)))
            hd5_file.create_dataset(HDF5DataSource.COVARIATES_KEY, shape=(generator_steps, len(column_names)),
                                    dtype=np.float32)
            string_dt = h5py.special_dtype(vlen=str)
            hd5_file.create_dataset(HDF5DataSource.ROWNAMES_KEY, shape=(generator_steps,), dtype=string_dt)

            for i, (row_name, array) in enumerate(generator):
                hd5_file[HDF5DataSource.ROWNAMES_KEY][i] = row_name
                hd5_file[HDF5DataSource.SHAPES_KEY][i] = array.shape
                hd5_file[HDF5DataSource.COVARIATES_KEY][i] = array.reshape((-1,))

            HDF5Tools._create_h5py_metadata(hd5_file=hd5_file, column_names=column_names,
                                            dataset_name=dataset_name, dataset_version=dataset_version,
                                            column_code_lists=column_code_lists)

    @staticmethod
    def save_h5_file(target_h5_file_path: AnyStr, data: np.ndarray, dataset_name: AnyStr,
                     column_names: List[AnyStr] = None, row_names: List[AnyStr] = None,
                     dataset_version: AnyStr = "1", column_code_lists: List[Dict[int, AnyStr]] = None):
        """
        Save, in bulk, a hd5 file with the expected format for usage with __h5pyDataSource__. This operation may
        require a large amount of memory.

        Args:
            target_h5_file_path: The target hd5 file path to write to.
            data: The data to store.
            dataset_name: The dataset name.
            column_names: The column names.
            row_names: The row names.
            dataset_version: The dataset version.
            column_code_lists: The column code lists.

        Returns:

        """
        if column_names is None:
            column_names = list(map(str, range(data.shape[-1])))
        if row_names is None:
            row_names = list(map(str, range(len(data))))
        with h5py.File(target_h5_file_path, "w") as hd5_file:
            if len(data) > 0:
                is_array = isinstance(data[0], np.ndarray)
                if isinstance(data[0], six.string_types):
                    dt = str
                elif is_array:
                    dt = data[0].dtype
                if data.dtype == object:
                    if is_array and len(data[0].shape) > 1:
                        # Multidimensional variable size arrays are not supported by HD5py - flatten & store shape.
                        shapes = np.array(list(map(lambda x: x.shape, data)), dtype=np.int32)
                        data = np.array(list(map(lambda x: x.reshape((-1,)), data)))
                        hd5_file.create_dataset(HDF5DataSource.SHAPES_KEY, data=shapes)
                    dt = h5py.special_dtype(vlen=dt)
                    hd5_file.create_dataset(HDF5DataSource.COVARIATES_KEY, data=data, dtype=dt)
                else:
                    hd5_file.create_dataset(HDF5DataSource.COVARIATES_KEY, data=data)
            HDF5Tools._create_h5py_metadata(hd5_file=hd5_file, column_names=column_names, row_names=row_names,
                                            dataset_name=dataset_name, dataset_version=dataset_version,
                                            column_code_lists=column_code_lists)

    @staticmethod
    def _create_h5py_metadata(hd5_file: h5py.File,
                              column_names: List[AnyStr],
                              dataset_name: AnyStr, dataset_version: AnyStr,
                              column_code_lists: Optional[List[Dict[int, AnyStr]]] = None,
                              row_names: Optional[List[AnyStr]] = None):
        string_dt = h5py.special_dtype(vlen=str)
        if row_names is not None:
            hd5_file.create_dataset(HDF5DataSource.ROWNAMES_KEY,
                                    data=np.array(row_names, dtype=object), dtype=string_dt)
        hd5_file.create_dataset(HDF5DataSource.COLNAMES_KEY,
                                data=np.array(column_names, dtype=object), dtype=string_dt)

        if column_code_lists is not None:
            for idx, column_value_map in enumerate(column_code_lists):
                if column_value_map is not None:
                    keys, values = zip(*column_value_map.items())
                    if column_value_map is not None:
                        hd5_file.create_dataset(HDF5DataSource.COLUMN_CODE_LIST_KEYS_KEY.format(idx),
                                                data=np.array(keys, dtype=np.int32))
                        hd5_file.create_dataset(HDF5DataSource.COLUMN_CODE_LIST_VALUES_KEY.format(idx),
                                                data=np.array(values, dtype=object), dtype=string_dt)
        hd5_file.attrs[HDF5DataSource.DATASET_NAME] = dataset_name
        hd5_file.attrs[HDF5DataSource.DATASET_VERSION] = dataset_version

    @staticmethod
    def from_comma_separated_string(datasets: AnyStr) -> CompositeDataSource:
        """ Loads a composite data source from a comma-separated string of data sources. """
        data_sources = []
        for dataset_path in datasets.split(","):
            data_source = HDF5DataSource(dataset_path)
            data_sources.append(data_source)

        combined_data_source = CompositeDataSource(data_sources)
        return combined_data_source
