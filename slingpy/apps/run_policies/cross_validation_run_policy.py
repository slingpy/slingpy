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
from typing import AnyStr, List, Dict
from slingpy.apps.app_paths import AppPaths
from slingpy.utils.metric_dict_tools import MetricDictTools
from slingpy.apps.run_policies.composite_run_policy import CompositeRunPolicy
from slingpy.apps.run_policies.abstract_run_policy import AbstractRunPolicy, RunResult, RunResultWithMetaData


class CrossValidationRunPolicy(CompositeRunPolicy):
    """
    A run policy for cross validated runs. Can be nested to allow for nested cross validation.
    """
    def __init__(self, base_policy: AbstractRunPolicy, evaluate_against, app_paths: AppPaths,
                 cross_validation_name: AnyStr = "inner", run_parallel: bool = False,
                 max_num_parallel_runs: int = 10):
        super(CrossValidationRunPolicy, self).__init__(
            base_policy=base_policy, app_paths=app_paths,
            run_parallel=run_parallel, max_num_parallel_runs=max_num_parallel_runs
        )
        self.evaluate_against = evaluate_against
        """ The fold to evaluate against. One of ('val' or 'test'). """
        self.cross_validation_name = cross_validation_name
        """ Name of the cross validation runs. """

    def get_split_index_name(self):
        split_index_name = f"split_index_{self.cross_validation_name}"
        return split_index_name

    def prepare_arguments(self, original_arguments) -> List[Dict]:
        num_splits_inner = original_arguments[f"num_splits_{self.cross_validation_name}"]
        if num_splits_inner < 2:
            raise AssertionError(f"__num__splits_{self.cross_validation_name}__ must be at least 2.")
        if num_splits_inner == 2:
            new_kwargs = dict(original_arguments)
            new_kwargs["evaluate_against"] = self.evaluate_against
            return [new_kwargs]

        split_index_name = self.get_split_index_name()
        output_directory_before = original_arguments["output_directory"]

        all_kwargs = []
        for split_index_inner in range(num_splits_inner):
            new_kwargs = dict(original_arguments)
            new_kwargs[split_index_name] = split_index_inner
            new_kwargs["output_directory"] = os.path.join(output_directory_before,
                                                          f"{self.cross_validation_name}_{split_index_inner:d}")
            if not os.path.exists(new_kwargs["output_directory"]):
                os.mkdir(new_kwargs["output_directory"])
            all_kwargs.append(new_kwargs)
        return all_kwargs

    def aggregate_results(self, results: List[RunResultWithMetaData], original_arguments: Dict) -> RunResult:
        if len(results) == 0:
            raise AssertionError("Must have results to aggregate.")

        split_index_name = self.get_split_index_name()

        eval_scores, test_scores, model_paths = [], [], []
        for i, result_w_metadata in enumerate(results):
            eval_scores.append(result_w_metadata.run_result.validation_scores)
            test_scores.append(result_w_metadata.run_result.test_scores)
            model_paths.append(result_w_metadata.run_result.model_path)

        MetricDictTools.print_metric_statistics(
            test_scores, f"{self.cross_validation_name}_{results[-1].arguments[split_index_name]:d} cross validation"
        )
        eval_score, test_score = MetricDictTools.aggregate_metric_dicts(eval_scores, test_scores)
        output_directory = original_arguments["output_directory"]
        MetricDictTools.save_metric_dict(eval_score, self.app_paths.get_eval_score_dict_path(output_directory))
        MetricDictTools.save_metric_dict(test_score, self.app_paths.get_test_score_dict_path(output_directory))
        model_path = model_paths[0] if len(model_paths) > 0 else None
        return RunResult(validation_scores=eval_score, test_scores=test_score, model_path=model_path)
