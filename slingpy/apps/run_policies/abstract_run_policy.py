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
import six
import time
import pickle
import traceback
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from typing import Dict, Any, AnyStr, Tuple, Optional
from slingpy.utils.argument_dictionary import ArgumentDictionary


class TracebackException(Exception):
    def __init__(self, source_err, tb):
        self.source_err = source_err
        self.tb = tb

    def get_source_error(self):
        return self.source_err

    def get_traceback(self):
        return self.tb


@dataclass
class RunResult(object):
    validation_scores: Dict[AnyStr, Any]
    test_scores: Dict[AnyStr, Any]
    model_path: Optional[AnyStr] = None


@dataclass
class RunResultWithMetaData(object):
    run_result: RunResult
    run_time: float  # In seconds.
    arguments: Dict


@six.add_metaclass(ABCMeta)
class AbstractRunPolicy(ArgumentDictionary):
    """
    Abstract base class for runnable policies.
    """
    @abstractmethod
    def _run(self, **kwargs) -> RunResult:
        """
        Executes the run policy.

        Args:
            **kwargs: Run arguments.

        Returns:
            The run results.
        """
        raise NotImplementedError()

    def run(self, **kwargs) -> RunResultWithMetaData:
        run_start_time = time.time()
        run_results = self._run(**kwargs)
        run_time = time.time() - run_start_time

        run_results_w_metadata = RunResultWithMetaData(run_results, run_time, arguments=kwargs)
        return run_results_w_metadata

    def is_async_run_policy(self):
        """
        Check whether this run policy can be executed asynchronously.

        Returns:
            True if the run policy can run asynchronously.
        """
        return False

    @staticmethod
    def run_with_file_output(inputs: Tuple[int, Dict], base_policy: "AbstractRunPolicy",
                             is_parallel: bool = True) -> Tuple[int, AnyStr]:
        """
        Runnable wrapper static function for use with __functools.partial__ and pool executors.

        Args:
            inputs: The inputs consisting of the run index and arguments.
            base_policy: The base policy to execute.
            is_parallel: Whether this job is being executed in a subprocess (for exception capture).

        Returns:
            A tuple consisting of the run index and a filepath to the results file (as __RunResultWithMetaData__)
             written to disk. The run results are returned as a reference filepath since pool executors are not
             optimized for large serialised objects.
        """
        index, kwargs = inputs
        output_directory = kwargs["output_directory"]
        tmp_results_dir = os.path.join(output_directory, "run_results")
        if not os.path.isdir(tmp_results_dir):
            os.mkdir(tmp_results_dir)

        try:
            run_results_w_metadata = base_policy.run(**kwargs)
        except Exception as e:
            if is_parallel:
                # Propagate exceptions to the handler.
                run_results_w_metadata = TracebackException(e, traceback.format_exc())
            else:
                raise  # Re-raise if not in subprocess.

        tmp_result_file_path = os.path.join(tmp_results_dir, "results.pickle")
        with open(tmp_result_file_path, "wb") as fp:
            pickle.dump(run_results_w_metadata, fp)
        return index, tmp_result_file_path
