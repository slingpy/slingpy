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
from typing import Tuple, Dict, Union, List, Any, AnyStr
from slingpy.apps.run_policies.hyperopt.exploration import AbstractExplorationStrategy


class RandomExplorationStrategy(AbstractExplorationStrategy):
    """
    A random exploration strategy for hyper-parameter optimization.
    """
    def __init__(self, hyperopt_param_ranges: Dict[AnyStr, Union[List, Tuple]], seed: int = 909):
        super(RandomExplorationStrategy, self).__init__(hyperopt_param_ranges=hyperopt_param_ranges)
        self.random_state = np.random.RandomState(seed)

    def next(self, state: Any) -> Any:
        new_params = dict()
        for k, v in self.hyperopt_param_ranges.items():
            if isinstance(v, list):
                min_val, max_val = v
                new_params[k] = float(self.random_state.uniform(min_val, max_val))
            elif isinstance(v, tuple):
                choice = self.random_state.randint(len(v))
                new_params[k] = v[choice]
        return new_params, state
