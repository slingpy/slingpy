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
import hashlib
import numpy as np
from typing import List
from abc import ABCMeta, abstractmethod


@six.add_metaclass(ABCMeta)
class AbstractMergeStrategy(object):
    def __init__(self):
        super(AbstractMergeStrategy, self).__init__()

    def __hash__(self):
        m = hashlib.sha256()
        m.update(bytes(type(self).__name__, 'utf-8'))
        return int(m.hexdigest(), 16)

    def __eq__(self, other):
        return type(self).__name__ == type(other).__name__

    @abstractmethod
    def resolve(self, query_result: List[np.ndarray]) -> List[np.ndarray]:
        """
        Resolves a list of multiple query results for a single row name in a data source into a single representative
        return value.

        Returns:
            A resolved List with a single representative entry value.
        """
        raise NotImplementedError()