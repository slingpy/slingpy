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
import six
import sys
from abc import ABCMeta
from typing import AnyStr, Type
from slingpy.models.abstract_base_model import AbstractBaseModel


if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


@six.add_metaclass(ABCMeta)
class PickleableBaseModel(AbstractBaseModel):
    @classmethod
    def load(cls: Type[AbstractBaseModel], file_path: AnyStr) -> "AbstractBaseModel":
        with open(file_path, "rb") as load_file:
            return pickle.load(load_file)

    @staticmethod
    def get_save_file_extension() -> AnyStr:
        return "pickle"

    def save(self, file_path: AnyStr):
        with open(file_path, "wb") as save_file:
            pickle.dump(self, save_file, pickle.HIGHEST_PROTOCOL)
