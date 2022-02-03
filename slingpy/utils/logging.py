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
from __future__ import print_function

import sys
import time


def error(*msg, **kwargs):
    log(*msg, log_level="ERROR", **kwargs)


def warn(*msg, **kwargs):
    log(*msg, log_level="WARN", **kwargs)


def info(*msg, **kwargs):
    log(*msg, log_level="INFO", **kwargs)


def log(*msg, **kwargs):
    sep = kwargs.pop('sep', " ")
    end = kwargs.pop('end', "\n")
    log_level = kwargs.pop('log_level', "INFO")
    with_timestamp = kwargs.pop('with_timestamp', True)

    initial_sep = " " if sep == "" else ""
    timestamp = " [{:.7f}]".format(time.time()) if with_timestamp else ""

    print(log_level + timestamp + ":" + initial_sep, *msg, sep=sep, end=end,
          file=sys.stdout if log_level == "INFO" else sys.stderr)


def report_duration(task, duration):
    log(task, "took", duration, "seconds.")


def time_function(task_name):
    def time_function(func):
        def func_wrapper(*args, **kargs):
            t_start = time.time()
            return_value = func(*args, **kargs)
            t_dur = time.time() - t_start
            report_duration(task_name, t_dur)
            return return_value
        return func_wrapper
    return time_function
