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
from sklearn.base import BaseEstimator
from typing import Optional, List, AnyStr
from slingpy.utils.plugin_tools import PluginTools
from slingpy.data_access.merge_strategies import *
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.models.pickleable_base_model import PickleableBaseModel
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


class SklearnModel(PickleableBaseModel):
    def __init__(self, base_module: BaseEstimator,
                 merge_strategy_x: AbstractMergeStrategy = NoMergeStrategy(),
                 merge_strategy_y: AbstractMergeStrategy = NoMergeStrategy()):
        super(SklearnModel, self).__init__()
        self.base_module = base_module
        self.merge_strategy_x = merge_strategy_x
        self.merge_strategy_y = merge_strategy_y
        self.model = None

    def build(self) -> BaseEstimator:
        available_model_params = PluginTools.get_available_instance_parameters(
            self.base_module, self.base_module.get_params()
        )
        return self.base_module.__class__(**available_model_params)

    def get_model_prediction(self, data):
        if hasattr(self.model, "predict_proba"):
            y_pred = self.model.predict_proba(data)
        else:
            y_pred = self.model.predict(data)
        return y_pred

    def predict(self, dataset_x: AbstractDataSource, batch_size: int = 256,
                row_names: List[AnyStr] = None) -> List[np.ndarray]:
        if self.model is None:
            self.model = self.build()
        if row_names is None:
            row_names = dataset_x.get_row_names()

        all_ids, y_preds, y_trues = [], [], []
        while len(row_names) > 0:
            current_indices = row_names[:batch_size]
            row_names = row_names[batch_size:]
            data = dataset_x.get_data(current_indices)
            data = self.merge_strategy_x.resolve(data)[0]
            y_pred = self.get_model_prediction(data)
            y_preds.append(y_pred)

        if len(y_preds) != 0:
            y_preds = np.concatenate(y_preds, axis=0)
            return [y_preds]
        else:
            return []

    def fit(self, train_x: AbstractDataSource, train_y: Optional[AbstractDataSource] = None,
            validation_set_x: Optional[AbstractDataSource] = None,
            validation_set_y: Optional[AbstractDataSource] = None) -> "AbstractBaseModel":
        if self.model is None:
            self.model = self.build()

        label_row_names = train_y.get_row_names()

        x, y = self.merge_strategy_x.resolve(train_x.get_data(label_row_names))[0], None
        if train_y is not None:
            y = self.merge_strategy_y.resolve(train_y.get_data(label_row_names))[0]
        self.model.fit(x, y=y)
        return self
