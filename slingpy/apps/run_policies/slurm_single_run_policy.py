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
from __future__ import print_function

import os
import sys
import pickle
from slingpy.apps.app_paths import AppPaths
from slingpy.schedulers.slurm_scheduler import SlurmScheduler
from slingpy.apps.run_policies.abstract_run_policy import AbstractRunPolicy, RunResult


class SlurmSingleRunPolicy(AbstractRunPolicy):
    """
    A runnable policy for runs executed remotely via slurm --wrap. The executable __base_policy__ must have a __main__
    entry point. Dependencies are loaded via a virtualenv that must be defined in
    __base_policy.remote_execution_virtualenv_path__.
    """
    def __init__(self, base_policy: "AbstractBaseApplication", app_paths: AppPaths):
        self.app_paths = app_paths
        """ A reference to the application paths object. """
        self.base_policy = base_policy
        """ The base policy to run. """

    def _run(self, **kwargs) -> RunResult:
        time_limit_days = self.base_policy.remote_execution_time_limit_days
        time_limit_hours = self.base_policy.remote_execution_time_limit_hours
        mem_limit_in_mb = self.base_policy.remote_execution_mem_limit_in_mb
        num_cpus = self.base_policy.remote_execution_num_cpus
        virtualenv_path = self.base_policy.remote_execution_virtualenv_path

        output_directory = kwargs["output_directory"]

        kwargs["single_run"] = True
        contents, err_contents = SlurmScheduler.execute(self.base_policy.__class__,
                                                        time_limit_days=time_limit_days,
                                                        time_limit_hours=time_limit_hours,
                                                        num_cpus=num_cpus,
                                                        mem_limit_in_mb=mem_limit_in_mb,
                                                        virtualenv_path=virtualenv_path,
                                                        project_dir_path=self.app_paths.project_root_directory,
                                                        output_directory=output_directory,
                                                        program_arguments=kwargs)

        if os.path.isfile(contents):
            with open(contents, "r") as fp:
                for line in fp:
                    print(f"[SLURM] {line}", file=sys.stdout)

        if os.path.isfile(err_contents):
            with open(err_contents, "r") as fp:
                for line in fp:
                    print(f"[SLURM] {line}", file=sys.stderr)

        run_results_path = self.app_paths.get_run_results_path(output_directory)
        with open(run_results_path, "rb") as fp:
            run_result = pickle.load(fp)
            return run_result

    def is_async_run_policy(self):
        return True
