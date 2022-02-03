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
import numpy as np
from typing import Tuple
from sklearn.datasets import load_diabetes
from slingpy.data_access.data_sources.hdf5_tools import HDF5Tools
from slingpy.data_access.data_sources.hdf5_data_source import HDF5DataSource
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


class Diabetes(object):
    @staticmethod
    def load_data(save_directory) -> Tuple[AbstractDataSource, AbstractDataSource]:
        diabetes = load_diabetes()
        x, y = diabetes['data'], diabetes['target'][:, np.newaxis]
        feature_names = diabetes['feature_names']

        h5_file_x = os.path.join(save_directory, "diabetes_x.h5")
        h5_file_y = os.path.join(save_directory, "diabetes_y.h5")
        HDF5Tools.save_h5_file(h5_file_x, x, "diabetes_x", column_names=feature_names)
        HDF5Tools.save_h5_file(h5_file_y, y, "diabetes_y")

        data_source_x = HDF5DataSource(h5_file_x)
        data_source_y = HDF5DataSource(h5_file_y)
        return data_source_x, data_source_y
