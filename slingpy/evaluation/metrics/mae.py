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
from sklearn.metrics import mean_absolute_error
from slingpy.evaluation.metrics.abstract_metric import AbstractMetric


class MeanAbsoluteError(AbstractMetric):
    """
    Mean absolute error metric.
    """

    def get_abbreviation(self) -> AnyStr:
        return "MAE"

    def evaluate(
        self, y_pred: np.ndarray, y_true: np.ndarray, threshold: Optional[float] = None
    ) -> np.ndarray:
        if y_pred.shape[1] == 2:
            # we assume the prediction of the larger class are stored in the second column
            y_pred = y_pred[:, 1]
        return mean_absolute_error(y_true, y_pred)
