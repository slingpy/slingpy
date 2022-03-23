"""
Copyright (C) 2021  Patrick Schwab, Arash Mehrjou, GlaxoSmithKline plc; Andrew Jesson, University of Oxford; Ashkan Soleymani, MIT
Copyright (C) 2020  Patrick Schwab, F.Hoffmann-La Roche Ltd  (source: https://github.com/d909b/CovEWS)
Copyright (C) 2019  Patrick Schwab, ETH Zurich

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
import shutil
import pickle
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABCMeta, abstractmethod
from slingpy.apps.app_paths import AppPaths
from slingpy.utils.logging import info, warn
from slingpy.utils.path_tools import PathTools
from slingpy.utils.init_seeds import init_seeds
from slingpy.evaluation.evaluator import Evaluator
from slingpy.utils.auto_argparse import AutoArgparse
from slingpy.utils.metric_dict_tools import MetricDictTools
from slingpy.utils.gpu_tools import get_num_available_gpus
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.data_access.data_sources.hdf5_tools import HDF5Tools
from typing import Tuple, Dict, Union, List, AnyStr, Any, Optional
from slingpy.evaluation.metrics.abstract_metric import AbstractMetric
from slingpy.apps.run_policies.slurm_single_run_policy import SlurmSingleRunPolicy
from slingpy.apps.run_policies.local_single_run_policy import LocalSingleRunPolicy
from slingpy.apps.run_policies.hyperopt.hyperopt_run_policy import HyperoptRunPolicy
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource
from slingpy.apps.run_policies.abstract_run_policy import AbstractRunPolicy, RunResult
from slingpy.apps.run_policies.cross_validation_run_policy import CrossValidationRunPolicy
from slingpy.apps.run_policies.hyperopt.exploration import RandomExplorationStrategy, SequentialExplorationStrategy


class DatasetHolder(object):
    pass


@six.add_metaclass(ABCMeta)
class AbstractBaseApplication(AbstractRunPolicy):
    """
    Abstract base application to derive a machine learning starter project from.
    """
    def __init__(self,
                 output_directory: AnyStr = "",
                 project_root_directory: AnyStr = "",
                 seed: int = 909,
                 evaluate: bool = True,
                 evaluate_against: AnyStr = "test",
                 hyperopt: bool = False,
                 hyperopt_metric_name: AnyStr = "MeanAbsoluteError",
                 hyperopt_comparison_operator: AnyStr = "<",  # One of '<' or '>'.
                 num_hyperopt_runs: int = 30,
                 split_index_outer: int = 1,
                 split_index_inner: int = 1,
                 num_splits_outer: int = 5,
                 num_splits_inner: int = 5,
                 save_predictions_batch_size: int = 256,
                 run_parallel_subtasks: bool = False,
                 single_run: bool = False,
                 schedule_on_slurm: bool = False,
                 save_predictions: bool = False,
                 save_predictions_file_format: AnyStr = "tsv",
                 nested_cross_validation: bool = True,
                 version_string: AnyStr = "0x000",
                 remote_execution_time_limit_days: int = 1,
                 remote_execution_time_limit_hours: int = 0,
                 remote_execution_mem_limit_in_mb: int = 2048,
                 remote_execution_num_cpus: int = 1,
                 remote_execution_virtualenv_path: AnyStr = ""):
        super(AbstractBaseApplication, self).__init__()
        self.seed = seed
        """ The random seed to use. """
        self.evaluate = evaluate
        """ Whether or not to evaluate in this run. """
        self.hyperopt = hyperopt
        """ Whether or not to run hyper-parameter optimization. """
        self.hyperopt_metric_name = hyperopt_metric_name
        self.hyperopt_comparison_operator = hyperopt_comparison_operator
        self.run_parallel_subtasks = run_parallel_subtasks
        self.single_run = single_run
        """ Whether or not to perform a single run only, i.e. disregarding any run policies in place. """
        self.version_string = version_string
        """ The version string corresponding to the current code version running. """
        self.evaluate_against = evaluate_against
        """ The dataset to evaluate against. One of ('val', 'test') """
        self.save_predictions_batch_size = save_predictions_batch_size
        """ Batch size to use when evaluating model for prediction output file saving. """

        self.is_temp_output_directory = output_directory == ""
        if self.is_temp_output_directory:
            output_directory = tempfile.mkdtemp()

        self.output_directory = output_directory
        """ 
        The output directory to write results to. If no __output_directory__ is specified, a temporary output directory
        is created to which outputs are written to. The temporary directory is automatically deleted after the run.
        You must specify an __output_directory__ to persist outputs.
        """
        self.save_predictions = save_predictions
        """ Whether or not to save model predictions to disk. """
        self.save_predictions_file_format = save_predictions_file_format
        """ File format to save predictions to. One of ('tsv', 'h5'). """
        self.num_hyperopt_runs = num_hyperopt_runs
        """ The number of hyper-parameter optimization runs to execute. """
        self.num_splits_outer = num_splits_outer
        self.num_splits_inner = num_splits_inner
        self.split_index_outer = split_index_outer
        self.split_index_inner = split_index_inner
        self.schedule_on_slurm = schedule_on_slurm
        """ Whether or not the code should be scheduled to run on slurm. """
        self.project_root_directory = project_root_directory
        """ The project root directory containing all custom project code that is not loaded via environment 
        dependencies. """
        self.nested_cross_validation = nested_cross_validation
        """ Whether or not to run nested cross validation. If __False__, cross validation with only one layer is 
        used. """
        self.remote_execution_num_cpus = remote_execution_num_cpus
        """ The number of CPUs to request for remote execution. Disregarded if __schedule_on_slurm__ is set to 
        false. """
        self.remote_execution_mem_limit_in_mb = remote_execution_mem_limit_in_mb
        """ The memory (in MB) to rquest for remote execution. Disregarded if __schedule_on_slurm__ is set to 
        false. """
        self.remote_execution_virtualenv_path = remote_execution_virtualenv_path
        """ The virtualenv path to load an environment from for remote execution. Disregarded if __schedule_on_slurm__ 
        is set to false. """
        self.remote_execution_time_limit_days = remote_execution_time_limit_days
        """ The execution time limit for remote execution in days. __remote_execution_time_limit_hours__ and
        __remote_execution_time_limit_days__ are added together to calculate the final execution time limit. """
        self.remote_execution_time_limit_hours = remote_execution_time_limit_hours
        """ The execution time limit for remote execution in hours.  __remote_execution_time_limit_hours__ and
        __remote_execution_time_limit_days__ are added together to calculate the final execution time limit. """

        info("Args are:", self.get_params())
        info("Running version", self.version_string)
        info("Running at", str(datetime.now()))

        self.datasets = None
        self.app_paths = self.get_app_paths()
        self.run_policy = self.get_run_policy()
        self.setup()

    def setup(self):
        """
        Setup the environment new machine learning starter project, e.g. by initializing random seeds.
        """
        info("There are", get_num_available_gpus(), "GPUs available.")
        init_seeds(int(np.rint(self.seed)))
        PathTools.mkdir_if_not_exists(self.output_directory)

    def get_app_paths(self) -> AppPaths:
        return AppPaths(self.project_root_directory)

    def get_run_policy(self) -> AbstractRunPolicy:
        """
        Get the active run policy, taking into account the current application configuration.
        """
        evaluate_against = self.evaluate_against

        if self.schedule_on_slurm:
            base_run = SlurmSingleRunPolicy(self, app_paths=self.app_paths)
        else:
            base_run = LocalSingleRunPolicy(self.run_single)

        cv_inner = CrossValidationRunPolicy(
            base_run,
            app_paths=self.app_paths,
            evaluate_against=evaluate_against,
            cross_validation_name="inner",
            run_parallel=self.run_parallel_subtasks
        )
        if self.single_run:
            return base_run
        elif self.hyperopt:
            num_hyperopt_runs = int(np.rint(self.num_hyperopt_runs))
            hyperopt_params = self.get_hyperopt_parameter_ranges()
            max_permutations = HyperoptRunPolicy.calculate_num_hyperopt_permutations(hyperopt_params)
            max_num_hyperopt_runs = int(min(max_permutations, num_hyperopt_runs))
            enumerate_all_permutations = max_permutations <= num_hyperopt_runs
            if enumerate_all_permutations:
                exploration_strategy = SequentialExplorationStrategy(hyperopt_params)
            else:
                exploration_strategy = RandomExplorationStrategy(hyperopt_params)

            hyperopt_policy = HyperoptRunPolicy(
                app_paths=self.app_paths,
                base_policy=cv_inner,
                hyperopt_params=self.get_hyperopt_parameter_ranges(),
                exploration_strategy=exploration_strategy,
                run_parallel=self.run_parallel_subtasks,
                max_num_hyperopt_runs=max_num_hyperopt_runs   # Do not perform more runs than necessary.
            )
            cv_outer = CrossValidationRunPolicy(
                hyperopt_policy,
                app_paths=self.app_paths,
                evaluate_against=evaluate_against,
                run_parallel=self.run_parallel_subtasks,
                cross_validation_name="outer"
            )
            return cv_outer
        else:
            if self.nested_cross_validation:
                cv_outer = CrossValidationRunPolicy(
                    cv_inner,
                    app_paths=self.app_paths,
                    evaluate_against=evaluate_against,
                    cross_validation_name="outer"
                )
                return cv_outer
            else:
                return cv_inner

    def get_hyperopt_parameter_ranges(self) -> Dict[AnyStr, Union[List, Tuple]]:
        """
        Get hyper-parameter optimization ranges.

        Returns:
            A dictionary with each item corresponding to a named hyper-parameter and its associated discrete
            (represented as a Tuple) or continuous (represented as a List[start, end]) value range.
        """
        return {}

    @abstractmethod
    def load_data(self) -> Dict[AnyStr, AbstractDataSource]:
        """
        Loads data sources for model training and/or evaluation.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_model(self) -> AbstractBaseModel:

        """
        Instantiate a machine learning model.

        Returns:
            An instantiated base model.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_model(self, model: AbstractBaseModel) -> Optional[AbstractBaseModel]:
        """
        Train a machine learning model.

        Args:
            model: The model to be trained.

        Returns:
            A trained base model and the training meta data.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_metrics(self, set_name: AnyStr) -> List[AbstractMetric]:
        raise NotImplementedError()

    def get_save_prediction_dataset_names(self) -> List[AnyStr]:
        """ Obtain names of datasets for which you would like to save predictions. """
        return ["training_set_x", "validation_set_x", "test_set_x"]

    def save_predictions_with_format(self, file_path, data, row_names, column_names, dataset_name):
        if self.save_predictions_file_format == "h5":
            HDF5Tools.save_h5_file(file_path, data=data, row_names=row_names, column_names=column_names,
                                   dataset_name=dataset_name, dataset_version=self.version_string)
        elif self.save_predictions_file_format == "tsv":
            df = pd.DataFrame(data, columns=column_names, index=row_names)
            df.index.name = "id"
            df.to_csv(file_path, sep="\t")
        else:
            raise AssertionError(f"Unsupported file format {self.save_predictions_file_format}.")

    def run_save_predictions(self, model, threshold: Optional[float] = None):
        info("Saving model predictions.")

        for dataset_name in self.get_save_prediction_dataset_names():
            if not hasattr(self.datasets, dataset_name):
                warn(f"Could not save predictions for dataset {dataset_name} because it was not loaded. Skipping.")
                continue

            dataset = getattr(self.datasets, dataset_name)
            if not isinstance(dataset, AbstractDataSource):
                warn(f"Could not save predictions for dataset {dataset_name} because it was not an "
                     f"AbstractDataSource instance. Skipping.")
                continue

            row_names = dataset.get_row_names()
            if len(row_names) == 0:
                continue

            outputs = model.predict(dataset, batch_size=self.save_predictions_batch_size)

            for output_idx in range(len(outputs)):
                output_i = outputs[output_idx]
                num_predictions = 1 if len(output_i.shape) == 1 else output_i.shape[-1]
                output_names = [f"label_{idx}" for idx in range(num_predictions)]
                assert num_predictions == len(output_names)

                file_path = self.get_prediction_file_path(dataset_name,
                                                          output_index=output_idx,
                                                          extension=self.save_predictions_file_format)
                try:
                    self.save_predictions_with_format(file_path, data=output_i, row_names=row_names,
                                                      column_names=output_names, dataset_name=dataset_name)
                    info("Saved raw model predictions to", file_path)

                    if threshold is not None:
                        thresholded_file_path = self.get_thresholded_prediction_file_path(
                            dataset_name, extension=self.save_predictions_file_format, output_index=output_idx
                        )
                        self.save_predictions_with_format(thresholded_file_path,
                                                          data=(output_i > threshold).astype(int),
                                                          row_names=row_names,
                                                          column_names=output_names,
                                                          dataset_name=dataset_name)
                        info("Saved thresholded model predictions to", thresholded_file_path)
                except AssertionError:
                    warn(f"Could not save prediction file of type {self.save_predictions_file_format} because it is not "
                         f"a supported format. Must be one of ('tsv', 'h5'). Skipping.")
                    continue

    def run_evaluation(self, model: AbstractBaseModel) -> Tuple[Dict[AnyStr, Any], Dict[AnyStr, Any], Optional[float]]:
        """
        Run all model evaluation steps to evaluate the performance of __model__.

        Args:
            model: The model whose performance is to be estimated.

        Returns:
            Metric dictionaries corresponding to the in-sample and out-of-sample scores and a threshold for prediction
            discretization in classification settings.
        """
        threshold = None
        evaluate_against = self.evaluate_against
        if evaluate_against not in ("test", "val"):
            warn(f"WARN: Specified wrong argument for --evaluate_against. Value was: {evaluate_against}. "
                 f"Defaulting to: val.")
            evaluate_against = "val"

        if evaluate_against == "test":
            thres_set_x = self.datasets.validation_set_x
            thres_set_y = self.datasets.validation_set_y
            eval_set_x = self.datasets.test_set_x
            eval_set_y = self.datasets.test_set_y
        else:
            thres_set_x = self.datasets.training_set_x
            thres_set_y = self.datasets.training_set_y
            eval_set_x = self.datasets.validation_set_x
            eval_set_y = self.datasets.validation_set_y

        # Get threshold from train or validation set.
        thres_score = self.evaluate_model(model, thres_set_x, thres_set_y, with_print=False, set_name="threshold")
        if "threshold" in thres_score:
            threshold = thres_score["threshold"]

        eval_score = self.evaluate_model(model, eval_set_x, eval_set_y, set_name=evaluate_against, threshold=threshold)

        if self.evaluate:
            if eval_score is None:
                test_score = self.evaluate_model(model, self.datasets.test_set_x, self.datasets.test_set_y,
                                                 with_print=evaluate_against == "val", set_name="test")
                eval_score = test_score
            else:
                test_score = eval_score
                eval_score = thres_score
        else:
            test_score = None
        return eval_score, test_score, threshold

    def evaluate_model(self, model: AbstractBaseModel, dataset_x: AbstractDataSource, dataset_y: AbstractDataSource,
                       with_print: bool = True, set_name: AnyStr = "", threshold=None) \
            -> Dict[AnyStr, Union[float, List[float]]]:
        """
        Evaluates model performance.

        Args:
            model: The model to evaluate.
            dataset: The dataset used for evaluation.
            with_print: Whether or not to print results to stdout.
            set_name: The name of the dataset being evaluated.
            threshold: An evaluation threshold (derived in sample) for discrete classification metrics,
             or none if a threshold should automatically be selected.

        Returns:
            A dictionary with each entry corresponding to an evaluation metric with one or more associated values.
        """
        return Evaluator.evaluate(model, dataset_x, dataset_y, self.get_metrics(set_name),
                                  with_print=with_print, set_name=set_name, threshold=threshold)

    def get_model_folder_path(self) -> AnyStr:
        return self.app_paths.get_model_folder_path(output_directory=self.output_directory)

    def get_model_file_path(self, extension: AnyStr = None) -> AnyStr:
        if extension is None:
            extension = self.get_model().get_save_file_extension()
        return self.app_paths.get_model_file_path(output_directory=self.output_directory,
                                                  extension=extension)

    def get_prediction_file_path(self, set_name: AnyStr, extension: AnyStr, output_index: int) -> AnyStr:
        return self.app_paths.get_prediction_file_path(
            output_directory=self.output_directory, set_name=set_name, extension=extension, output_index=output_index
        )

    def get_thresholded_prediction_file_path(self, set_name: AnyStr, extension: AnyStr, output_index: int) -> AnyStr:
        return self.app_paths.get_thresholded_prediction_file_path(
            output_directory=self.output_directory, set_name=set_name, extension=extension, output_index=output_index
        )

    @staticmethod
    def instantiate_from_command_line(clazz):
        return AutoArgparse.instantiate_from_command_line(clazz)

    def cleanup(self):
        if self.is_temp_output_directory:
            info(f"Deleting output directory at {self.output_directory} because it was a temporary directory.")
            shutil.rmtree(self.output_directory)

    def _run(self, **kwargs) -> RunResult:
        run_result = self.run_policy._run(**self.get_params())
        self.cleanup()
        return run_result

    def init_data(self):
        if not self.datasets:
            datasets = self.load_data()
            self.datasets = DatasetHolder()
            for name, data_source in datasets.items():
                setattr(self.datasets, name, data_source)

    def run_single(self, **kwargs) -> RunResult:
        """ Executes a single run directly (disregarding any __self.run_policy__ that may be in place. """
        prior_args = kwargs
        self.set_params(**kwargs)

        info("Run with args:", self.get_params())

        output_directory = self.output_directory

        self.init_data()
        model = self.get_model()
        model = self.train_model(model)
        if model is not None:
            model_path = self.get_model_file_path(extension=model.get_save_file_extension())
            model.save(model_path)
        else:
            model_path = None

        args_file_path = self.app_paths.get_args_file_path(output_directory)

        info("Saving args to", args_file_path)
        with open(args_file_path, "wb") as fp:
            pickle.dump(self.get_params(), fp, pickle.HIGHEST_PROTOCOL)

        eval_score, test_score, threshold = None, None, None
        if self.evaluate:
            eval_score, test_score, threshold = self.run_evaluation(model)

        if self.save_predictions:
            self.run_save_predictions(model, threshold=threshold)

        MetricDictTools.save_metric_dict(eval_score, self.app_paths.get_eval_score_dict_path(output_directory))
        MetricDictTools.save_metric_dict(test_score, self.app_paths.get_test_score_dict_path(output_directory))
        self.set_params(**prior_args)
        result = RunResult(validation_scores=eval_score, test_scores=test_score, model_path=model_path)
        results_file_path = self.app_paths.get_run_results_path(output_directory)
        with open(results_file_path, "wb") as fp:
            pickle.dump(result, fp, pickle.HIGHEST_PROTOCOL)
        return result


if __name__ == "__main__":
    app = AbstractBaseApplication.instantiate_from_command_line(AbstractBaseApplication)
    app.run()
