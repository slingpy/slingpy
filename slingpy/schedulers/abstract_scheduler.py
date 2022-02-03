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
import sys
import six
import subprocess
from argparse import ArgumentParser
from slingpy.utils.logging import info
from abc import ABCMeta, abstractmethod
from typing import Set, Optional, AnyStr, Dict, Tuple


@six.add_metaclass(ABCMeta)
class AbstractScheduler(object):
    """
    An abstract scheduler for executing runnable code programmatically.
    """

    @staticmethod
    def run_command(cmd: AnyStr, logfile: AnyStr, err_logfile: AnyStr) -> Tuple[AnyStr, AnyStr]:
        info(f"Running command: {cmd}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        process.communicate()[0].decode("utf-8").strip()  # Propagate outputs to parent process (may be out of order).

        contents = ""
        if os.path.isfile(logfile):
            with open(logfile, "r") as fp:
                contents = fp.read()
        err_contents = ""
        if os.path.isfile(err_logfile):
            with open(err_logfile, "r") as fp:
                err_contents = fp.read()
        return contents, err_contents

    @staticmethod
    def convert_arguments_dict_to_program_argument_string(
            parser: ArgumentParser, kwargs: Dict, exclude: Optional[Set] = None
    ) -> AnyStr:
        """
        Converts a dictionary of program arguments to an argument string that may be passed on command line.

        Args:
            parser: An ArgParser instance corresponding to the argument set available.
            kwargs: The argument dictionary.
            exclude: Set of parameters to be excluded from the program argument string.

        Returns:
            A str
        """
        if exclude is None:
            exclude = set()

        arguments = []
        for arg_name, arg_value in kwargs.items():
            if isinstance(arg_value, bool) and arg_value is False:
                if parser.get_default(arg_name) is not False:
                    arguments.append(f"--do_not_{arg_name}")
                else:
                    continue
            elif isinstance(arg_value, bool) and arg_value is True:
                if parser.get_default(arg_name) is not True and arg_name not in exclude:
                    arguments.append(f"--do_{arg_name}")
                else:
                    continue
            else:
                arguments.append(f"--{arg_name}='{arg_value}'")
        return " ".join(arguments)

    @staticmethod
    @abstractmethod
    def execute(clazz, mem_limit_in_mb: int = 2048, num_cpus: int = 1, virtualenv_path: AnyStr = "",
                project_dir_path: AnyStr = "", exclude: AnyStr = "", **kwargs):
        raise NotImplementedError()
