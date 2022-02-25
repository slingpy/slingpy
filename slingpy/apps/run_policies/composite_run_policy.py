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
import pickle
from typing import List, Dict
from functools import partial
from abc import abstractmethod
from slingpy.utils.logging import error
from slingpy.apps.app_paths import AppPaths
from slingpy.utils.nestable_pool import NestablePool as Pool
from slingpy.apps.run_policies.abstract_run_policy import AbstractRunPolicy, RunResult, RunResultWithMetaData, \
    TracebackException


class CompositeRunPolicy(AbstractRunPolicy):
    """
    A composite run policy that executes multiple child policies.
    """
    def __init__(self, base_policy: AbstractRunPolicy, app_paths: AppPaths,
                 run_parallel: bool = False, max_num_parallel_runs: int = 10):
        self.app_paths = app_paths
        """ A reference to the application paths object. """
        self.base_policy = base_policy
        """ The base policy to execute in a cross validation setting. """
        self.run_parallel = run_parallel
        """ Whether or not to run the base policy in parallel asynchronous execution or sequentially. """
        self.max_num_parallel_runs = max_num_parallel_runs
        """ The maximum number of parallel runs that will be executed asynchronously. 
        Only has an effect if __run_parallel__ is set to True. """
        self.total_runtime_in_seconds = 0

    def is_async_run_policy(self):
        return self.base_policy.is_async_run_policy()

    @abstractmethod
    def prepare_arguments(self, original_arguments) -> List[Dict]:
        """
        Prepare arguments for cross validation runs.

        Args:
            original_arguments: The original run arguments.

        Returns:
            A prepared list of program arguments to run.
        """
        raise NotImplementedError()

    @abstractmethod
    def aggregate_results(self, results: List[RunResultWithMetaData], original_arguments: Dict) -> RunResult:
        """
        Aggregate the collection of run results for further processing.

        Args:
            results: A list of run results with associated meta data.
            original_arguments: The original program arguments.

        Returns:
            An aggregated run result.
        """
        raise NotImplementedError()

    def run(self, **kwargs) -> RunResultWithMetaData:
        run_results = self._run(**kwargs)
        run_time = self.total_runtime_in_seconds
        run_results_w_metadata = RunResultWithMetaData(run_results, run_time, arguments=kwargs)
        return run_results_w_metadata

    def handle_child_process_exceptions(self, arg_list, ordered_outputs):
        all_exceptions = []
        for args, outputs in zip(arg_list, ordered_outputs):
            if isinstance(outputs, TracebackException):
                error("Run args were:", args)
                error(outputs.get_traceback())
                error(outputs)
                all_exceptions.append([args, outputs])
        return all_exceptions

    def _run(self, **kwargs) -> RunResult:
        all_kwargs = self.prepare_arguments(kwargs)

        if self.base_policy.is_async_run_policy():
            num_processes = min(len(all_kwargs), self.max_num_parallel_runs)
        else:
            num_processes = 1

        if not self.run_parallel:
            result_dicts = list(map(
                partial(AbstractRunPolicy.run_with_file_output,
                        base_policy=self.base_policy,
                        is_parallel=self.run_parallel),
                zip(range(len(all_kwargs)), all_kwargs)
            ))
        else:
            with Pool(processes=num_processes) as pool:
                result_dicts = list(pool.imap_unordered(
                    partial(AbstractRunPolicy.run_with_file_output,
                            base_policy=self.base_policy,
                            is_parallel=self.run_parallel),
                    zip(range(len(all_kwargs)), all_kwargs),
                    chunksize=1
                ))
        result_dict_paths = list(map(lambda x: x[1], sorted(result_dicts, key=lambda x: x[0])))

        # Load the serialised results from disk.
        run_results = []
        for result_dict_path in result_dict_paths:
            with open(result_dict_path, "rb") as fp:
                run_results_w_metadata = pickle.load(fp)
                run_results.append(run_results_w_metadata)

        all_exceptions = self.handle_child_process_exceptions(all_kwargs, run_results)
        if len(all_exceptions) != 0:
            error_msg = f"There were {len(all_exceptions)} exceptions in subprocesses. Re-raising the last error."
            error(error_msg)
            raise Exception(error_msg) from all_exceptions[-1][1]

        for result_dict_path in result_dict_paths:
            # Remove references from disk after aggregation.
            os.unlink(result_dict_path)

        self.total_runtime_in_seconds = sum([run_result.run_time if hasattr(run_result, "run_time") else 0
                                             for run_result in run_results])
        run_result = self.aggregate_results(run_results, original_arguments=kwargs)
        return run_result
