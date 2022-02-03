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
from typing import Optional, AnyStr
from sklearn.metrics import mean_squared_error
from slingpy.evaluation.metrics.abstract_metric import AbstractMetric


class TopKRecall(AbstractMetric):
    """
    Recall in % of top K y_true in a regression setting.
    """

    def __init__(self, k: float = 0.1, top_percentile_threshold: float = 0.05):
        self.k = k
        self.top_percentile_threshold = top_percentile_threshold

    def get_abbreviation(self) -> AnyStr:
        return f"topk-recall_({self.k:f}, {self.top_percentile_threshold:f})"

    def evaluate(self, y_pred: np.ndarray, y_true: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        y_true_ranks, y_pred_ranks = np.argsort(y_true[:, 0]), np.argsort(y_pred[:, 0])
        y_true_rank_index = dict(zip(np.arange(len(y_true_ranks)), y_true_ranks))
        reverse_y_pred_rank_index = dict(zip(y_pred_ranks, np.arange(len(y_pred_ranks))))

        num_top = int(np.ceil(self.k * len(y_pred)))
        top_k = int(np.ceil(self.top_percentile_threshold * len(y_pred)))
        count = 0
        for k_i in range(num_top):
            count += (y_true_rank_index[reverse_y_pred_rank_index[k_i]] < top_k)
        ret_val = float(count) / top_k
        return np.array(ret_val)
