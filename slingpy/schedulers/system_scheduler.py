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
import inspect
from typing import AnyStr
from slingpy.utils.auto_argparse import AutoArgparse
from slingpy.schedulers.abstract_scheduler import AbstractScheduler


class SystemScheduler(object):
    """
    A scheduler for executing code on the command line programmatically via automated conversion of instance arguments
    to command line parameters.
    """

    @staticmethod
    def execute(clazz, mem_limit_in_mb: int = 2048, num_cpus: int = 1, virtualenv_path: AnyStr = "",
                project_dir_path: AnyStr = "", exclude: AnyStr = "", **kwargs):
        output_directory = kwargs["output_directory"]
        logfile = os.path.join(output_directory, "log.txt")
        err_logfile = os.path.join(output_directory, "errlog.txt")
        main_app_path = os.path.abspath(inspect.getfile(clazz))

        parser = AutoArgparse.get_parser_for_clazz(clazz)
        argument_list = AbstractScheduler.convert_arguments_dict_to_program_argument_string(parser, kwargs)
        cmd = f"source {virtualenv_path}/bin/activate && " \
              f"export PYTHONPATH={project_dir_path}:\$PYTHONPATH && " \
              f"export HDF5_USE_FILE_LOCKING='FALSE' && " \
              f"python {main_app_path} {argument_list}"
        cmd = f"{cmd} > {logfile} 2> {err_logfile} "

        contents, err_contents = AbstractScheduler.run_command(cmd, logfile, err_logfile)
        return contents, err_contents
