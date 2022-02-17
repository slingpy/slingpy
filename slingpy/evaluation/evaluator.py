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
from typing import Optional, List
from collections import OrderedDict
from slingpy.utils.logging import info, warn
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.evaluation.metrics.abstract_metric import AbstractMetric
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


class Evaluator(object):
    @staticmethod
    def evaluate(model: AbstractBaseModel, dataset_x: AbstractDataSource, dataset_y: AbstractDataSource,
                 metrics: List[AbstractMetric], set_name="Test set", with_print=True, threshold: Optional[float] = None,
                 batch_size: int = 256):
        y_pred = model.predict(dataset_x, batch_size=batch_size, row_names=dataset_y.get_row_names())
        y_true = dataset_y.get_data()

        assert len(y_pred) == len(y_true), "The evaluated model must have same number as outputs as in the target data."

        all_metric_dict = {}
        for output_idx in range(len(y_pred)):
            y_pred_i, y_true_i = y_pred[output_idx], y_true[output_idx]
            if len(y_pred_i.shape) == 1:
                y_pred_i = y_pred_i[:, np.newaxis]

            assert y_true_i.shape[0] == y_pred_i.shape[0]

            metric_dict, offset = OrderedDict(), 1
            for metric in metrics:
                metric_name = metric.__class__.__name__
                value = metric.evaluate(y_pred_i, y_true_i, threshold=threshold)
                if metric_name in metric_dict:
                    message = f"{metric_name} was already present in metric dict. " \
                              f"Do you have two metrics set for evaluation with the same name?"
                    warn(message)
                    raise AssertionError(message)
                metric_dict[metric_name] = value
            all_metric_dict[output_idx] = metric_dict

        if len(all_metric_dict) == 1:
            # Flatten metric dict if there is exactly one model output.
            all_metric_dict = all_metric_dict[0]

        if with_print:
            info(f"Performance on {set_name}", all_metric_dict)
        return all_metric_dict
