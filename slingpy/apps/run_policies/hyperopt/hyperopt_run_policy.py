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
import shutil
import numpy as np
from distutils.dir_util import copy_tree
from slingpy.apps.app_paths import AppPaths
from slingpy.utils.logging import info, warn
from slingpy.utils.path_tools import PathTools
from typing import Tuple, Dict, Union, List, AnyStr
from slingpy.utils.metric_dict_tools import MetricDictTools
from slingpy.apps.run_policies.composite_run_policy import CompositeRunPolicy
from slingpy.apps.run_policies.hyperopt.exploration import AbstractExplorationStrategy
from slingpy.apps.run_policies.abstract_run_policy import AbstractRunPolicy, RunResult, RunResultWithMetaData


class HyperoptRunPolicy(CompositeRunPolicy):
    """
    A runnable policy for hyper-parameter optimisation.
    """
    def __init__(self, base_policy: AbstractRunPolicy, app_paths: AppPaths,
                 hyperopt_params: Dict[AnyStr, Union[List, Tuple]],
                 exploration_strategy: AbstractExplorationStrategy,
                 max_num_hyperopt_runs: int, run_parallel: bool = False,
                 max_num_parallel_runs: int = 10):
        super(HyperoptRunPolicy, self).__init__(
            base_policy=base_policy, app_paths=app_paths,
            run_parallel=run_parallel, max_num_parallel_runs=max_num_parallel_runs
        )
        self.hyperopt_params = hyperopt_params
        """ The hyperopt parameters. """
        self.exploration_strategy = exploration_strategy
        """ The exploration strategy used. """
        self.max_num_hyperopt_runs = max_num_hyperopt_runs
        """ Maximum number of hyper-parameter optimization trials to run. 
        Note fewer trials are run if the number of all possible combinations of hyper-parameters is smaller than
        __self.max_num_hyperopt_runs__. """
        self.best_score_index = 0
        """ The index of the hyper-parameter optimization run that had the best score so far. """
        self.best_score = None
        """ The best observed score so far across hyper-parameter optimization runs. """
        self.best_params = ""
        """ The parameter set corresponding to the trial with the best observed score so far. """

    @staticmethod
    def calculate_num_hyperopt_permutations(hyperopt_parameters: Dict[AnyStr, Union[Tuple, List]]) -> float:
        """
        Calculate the maximum number of hyper-parameter optimization permutations.

        Args:
            hyperopt_parameters: A set of pre-defined hyper-parameter ranges. Every entry is a named hyper-parameter.

        Returns:
            The maximum number of hyper-parameter optimization permutations.
        """
        num_permutations = 1.0
        for param_range in hyperopt_parameters.values():
            if isinstance(param_range, list):
                return float("inf")
            else:
                num_permutations *= len(param_range)
        return num_permutations

    @staticmethod
    def print_run_results(args, hyperopt_parameters, run_index, score, run_time) -> AnyStr:
        """
        Print the results of a single hyper-parameter optimization trial.

        Args:
            args: The run arguments.
            hyperopt_parameters: The hyper-parameters.
            run_index: The index of the hyperopt trial run.
            score: The observed score.
            run_time: The run time of the hyperopt trial run (in seconds).

        Returns:
            The printed message containing the run details.
        """
        message = "Hyperopt run [" + str(run_index) + "]:"
        best_params_message = ""
        for k in hyperopt_parameters:
            best_params_message += k + "=" + "{}".format(args[k]) + ", "
        best_params_message += "time={:.4f},".format(run_time) + "score={:.4f}".format(score)
        info(message, best_params_message)
        return best_params_message

    def get_final_output_directory(self, original_arguments) -> AnyStr:
        final_output_directory = original_arguments["output_directory"]
        return final_output_directory

    def get_best_output_directory(self, original_arguments) -> AnyStr:
        final_output_directory = self.get_final_output_directory(original_arguments)
        best_output_directory = os.path.join(final_output_directory, "best_run")
        return best_output_directory

    def prepare_arguments(self, original_arguments) -> List[Dict]:
        hyperopt_offset = int(np.rint(original_arguments["hyperopt_offset"]))

        info("Performing hyper-parameter optimisation with parameters:", self.hyperopt_params)

        final_output_directory = self.get_final_output_directory(original_arguments)
        best_output_directory = self.get_best_output_directory(original_arguments)
        PathTools.mkdir_if_not_exists(best_output_directory)

        all_kwargs = []
        state, job_ids, score_dicts, test_score_dicts, eval_dicts = None, [], [], [], []
        for i in range(self.max_num_hyperopt_runs):
            new_kwargs = dict(original_arguments)
            new_params, state = self.exploration_strategy.next(state=state)
            new_kwargs.update(new_params)

            current_output_directory = os.path.join(final_output_directory, f"hyperopt_{i}")
            new_kwargs["output_directory"] = current_output_directory
            PathTools.mkdir_if_not_exists(current_output_directory)

            if i < hyperopt_offset:
                # Skip until we reached the hyperopt offset.
                continue

            all_kwargs.append(new_kwargs)
        return all_kwargs

    def aggregate_results(self, results: List[RunResultWithMetaData], original_arguments: Dict) -> RunResult:
        hyperopt_comparator = ">"  # One of < and >
        if "hyperopt_comparison_operator" in original_arguments:
            hyperopt_comparator = original_arguments["hyperopt_comparison_operator"]
            if hyperopt_comparator not in ["<", ">"]:
                raise NotImplementedError(f"Hyperopt comparison operator {hyperopt_comparator} does not exist. "
                                          f"Should be one of '<' or '>'.")
        hyperopt_metric_name = original_arguments["hyperopt_metric_name"]
        best_output_directory = self.get_best_output_directory(original_arguments)
        final_output_directory = self.get_final_output_directory(original_arguments)

        self.best_score = float("-inf") if hyperopt_comparator == ">" else float("inf")
        score_dicts, test_score_dicts, model_paths = [], [], []
        for i, results_w_metadata in enumerate(results):
            score_dicts.append(results_w_metadata.run_result.validation_scores)
            test_score_dicts.append(results_w_metadata.run_result.test_scores)
            model_paths.append(results_w_metadata.run_result.model_path)

            eval_dict = results_w_metadata.run_result.validation_scores
            score = eval_dict[hyperopt_metric_name]

            best_params_message = HyperoptRunPolicy.print_run_results(results_w_metadata.arguments,
                                                                      self.hyperopt_params,
                                                                      i, score, results_w_metadata.run_time)
            current_output_directory = results_w_metadata.arguments["output_directory"]
            if ((hyperopt_comparator == ">" and score > self.best_score) or
                (hyperopt_comparator == "<" and score < self.best_score)) and original_arguments["train"]:
                self.best_score_index = i
                self.best_score = score
                self.best_params = best_params_message

                if os.path.isdir(current_output_directory):
                    copy_tree(current_output_directory, best_output_directory)
                else:
                    warn(f"Not moving {current_output_directory} to {best_output_directory} due to "
                         "unexpected or missing folder structure.")

            if os.path.isdir(current_output_directory):
                shutil.rmtree(current_output_directory)
            else:
                warn(f"Not removing {current_output_directory} due to unexpected or missing folder structure.")

        info("Best[", self.best_score_index, "] config was", self.best_params)
        info("Best_test_score:", test_score_dicts[self.best_score_index])

        if os.path.isdir(best_output_directory):
            copy_tree(best_output_directory, final_output_directory)
            shutil.rmtree(best_output_directory)  # Remove best directory.
        else:
            warn(f"Not moving {best_output_directory} to {final_output_directory} due to "
                 "unexpected or missing folder structure.")

        MetricDictTools.print_metric_statistics(score_dicts, "Hyperopt")

        # Override last score dicts with best.
        output_directory = original_arguments["output_directory"]
        MetricDictTools.save_metric_dict(score_dicts[self.best_score_index],
                                         self.app_paths.get_eval_score_dict_path(output_directory))
        MetricDictTools.save_metric_dict(test_score_dicts[self.best_score_index],
                                         self.app_paths.get_test_score_dict_path(output_directory))

        if len(score_dicts) != 0:
            root_size = len(best_output_directory.split(os.path.sep))
            path_postfix = model_paths[self.best_score_index].split(os.path.sep)[root_size:]
            path_postfix = os.path.sep.join(path_postfix)
            model_path = os.path.join(final_output_directory, path_postfix)
            return RunResult(validation_scores=score_dicts[self.best_score_index],
                             test_scores=test_score_dicts[self.best_score_index],
                             model_path=model_path)
        else:
            return RunResult({}, {}, model_path=None)
