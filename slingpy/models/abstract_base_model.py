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
from abc import ABCMeta, abstractmethod
from typing import Optional, AnyStr, Type, List
from slingpy.utils.argument_dictionary import ArgumentDictionary
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


@six.add_metaclass(ABCMeta)
class AbstractBaseModel(ArgumentDictionary):
    """ An abstract base machine learning model. """
    def __init__(self, inputs: Optional = None, outputs: Optional = None):
        self.inputs = inputs
        self.outputs = outputs

    @abstractmethod
    def predict(self, dataset_x: AbstractDataSource, batch_size: int = 256,
                row_names: List[AnyStr] = None) -> List[np.ndarray]:
        """
        Produces model predictions for a given dataset. Can be run in batched mode to limit peak memory usage.

        Args:
            dataset_x: The dataset co-variates upon which to predict. Shape [N, ...] where N = number of samples.
            batch_size: If len(dataset_x) is larger than __batch_size__ multiple batches will be evaluated to limit
                        peak memory usage.
            row_names: List of row names to query in dataset_x (Default: None, all dataset_x row names will be queried)

        Returns:
            Model predictions. List of arrays with shape [N, ...] where N = number of samples in __dataset_x__.
                               Every list entry corresponds to a model output (multiple outputs for one model possible).
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, train_x: AbstractDataSource, train_y: Optional[AbstractDataSource] = None,
            validation_set_x: Optional[AbstractDataSource] = None,
            validation_set_y: Optional[AbstractDataSource] = None) \
            -> "AbstractBaseModel":
        """
        Fits the model to a training dataset.

        Args:
            train_x: The training set co-variates.
            train_y: The training set labels (for supervised learning). Optional for unsupervised learning.
            validation_set_x: The validation set co-variates to use for model optimization (e.g., early stopping).
            validation_set_y: The validation set labels.

        Returns:
            A reference to the model instance. Not a copy.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls: Type["AbstractBaseModel"], file_path: AnyStr) -> "AbstractBaseModel":
        """

        Args:
            file_path:

        Returns:

        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_save_file_extension() -> AnyStr:
        raise NotImplementedError()

    @abstractmethod
    def save(self, file_path: AnyStr):
        """

        Args:
            file_path:

        Returns:

        """
        raise NotImplementedError()
