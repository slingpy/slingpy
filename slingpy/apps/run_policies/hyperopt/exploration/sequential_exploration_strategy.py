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
from typing import Tuple, Dict, Union, List, Any, AnyStr
from slingpy.apps.run_policies.hyperopt.exploration import AbstractExplorationStrategy


class SequentialExplorationStrategy(AbstractExplorationStrategy):
    """
    A sequential exploration strategy for hyper-parameter optimization.
    """
    def __init__(self, hyperopt_param_ranges: Dict[AnyStr, Union[List, Tuple]]):
        super(SequentialExplorationStrategy, self).__init__(hyperopt_param_ranges=hyperopt_param_ranges)

    def next(self, state: Any) -> Any:
        if state is None:
            state = [0 for _ in range(len(self.hyperopt_param_ranges))]

        new_params = dict()
        for k, state_i in zip(sorted(self.hyperopt_param_ranges.keys()), state):
            v = self.hyperopt_param_ranges[k]
            if isinstance(v, tuple):
                choice = v[state_i]
                if hasattr(choice, "item"):
                    choice = choice.item()
                new_params[k] = choice
            else:
                raise AssertionError("Only hyperopt_parameters with finite numbers of permutations can be used"
                                     "with __get_next_hyperopt_choice_generator__.")

        for i, key in enumerate(sorted(self.hyperopt_param_ranges.keys())):
            state[i] += 1
            if state[i] % len(self.hyperopt_param_ranges[key]) == 0:
                state[i] = 0
            else:
                break

        return new_params, state
