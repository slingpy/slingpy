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
from slingpy.utils.logging import warn


class PathTools(object):
    """ Tools for manipulating file paths. """
    @staticmethod
    def mkdir_if_not_exists(new_dir, raise_error_if_exists: bool = False):
        """
        Creates a directory at __new_dir__ if it does not yet exist.

        Args:
            new_dir: Path to the new directory.
            raise_error_if_exists: Whether or not to raise an error if the directory already exists.

        Returns:

        """
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        else:
            message = f"{new_dir} already existed. Its previous contents may be overwritten."
            if raise_error_if_exists:
                raise AssertionError(message)
            else:
                warn(message)
