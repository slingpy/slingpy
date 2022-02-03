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
import inspect
from typing import AnyStr
from argparse import ArgumentParser


class AutoArgparse(object):
    @staticmethod
    def get_parser_for_clazz(clazz):
        parser = ArgumentParser(description='Entry for {clazz_name:}.'.format(
            clazz_name=clazz.__class__.__name__
        ))
        parameters = inspect.signature(clazz.__init__).parameters.items()
        for i, (parameter_name, parameter) in enumerate(parameters):
            if i == 0:
                continue  # Skip __self__.
            kwargs = {}
            if parameter.default is not inspect.Parameter.empty:
                kwargs["default"] = parameter.default
            else:
                kwargs["required"] = True

            if parameter.annotation is bool or parameter.annotation == "bool":
                defaults = {f"{parameter_name}": parameter.default}
                parser.set_defaults(**defaults)

                if parameter.default:
                    prefix = "do_not"
                    store_action = "store_false"
                else:
                    prefix = "do"
                    store_action = "store_true"
                parser.add_argument(f"--{prefix}_{parameter_name}", dest=f"{parameter_name}", action=store_action)
            else:
                if parameter.annotation is AnyStr or parameter.annotation == "AnyStr":
                    type_hint = str
                else:
                    type_hint = parameter.annotation
                parser.add_argument("--{parameter_name:}".format(parameter_name=parameter_name), help="",
                                    type=type_hint, **kwargs)
        return parser

    @staticmethod
    def instantiate_from_command_line(clazz):
        parser = AutoArgparse.get_parser_for_clazz(clazz)
        args = vars(parser.parse_args())
        instance = clazz(**args)
        return instance
