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
from contextlib import contextmanager

from ilock import ILock

from slingpy.utils.logging import warn


class PathTools:
    """Tools for manipulating file paths."""

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

    @staticmethod
    @contextmanager
    def ilock_nothrow(key, lock_directory):
        """
        THIS LOCK ISN'T GUARANTEED TO BE EXCLUSIVE ON NFS FILESYSTEMS!
        File operations aren't immediately propagated, allowing a small window after
        locking where a parallel processes can also acquire the lock. This allows
        approx. 1 in 20 processes to enter a lock that is already held, if they all
        try to acquire the lock simultaneously.

        This wrapper suppresses the FileNotFoundErrors that occur when a contested lock
        is released.
        """
        lock = ILock(key, lock_directory=lock_directory)
        lock.__enter__()
        try:
            yield lock
        finally:
            try:
                lock.__exit__(None, None, None)
            except FileNotFoundError:
                # Lockfile is missing - ignore
                pass
