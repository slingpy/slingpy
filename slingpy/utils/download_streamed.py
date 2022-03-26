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
import uuid
import pathlib
import requests
from ilock import ILock


def download_streamed(
    download_url,
    local_save_file_path,
    chunk_size=2**20,
    skip_if_exists=True,
):
    file_path = pathlib.Path(local_save_file_path)
    local_file_directory = file_path.parent
    with ILock(f"download_streamed_{local_save_file_path}", lock_directory=local_file_directory):
        if skip_if_exists and os.path.exists(local_save_file_path):
            return

        with requests.get(download_url, stream=True) as request:
            request.raise_for_status()
            # Use a temp file for downloading so that even a SIGKILL'd process
            # won't leave a partial download with the destination filename.
            # Also make it unique, so that it can't be corrupted by parallel processes
            # that can't share locks (e.g. cluster jobs with local /tmp folders)
            tmp_file = str(local_save_file_path) + f".download{uuid.uuid4().hex}"

            try:
                with open(tmp_file, "wb") as fp:
                    for chunk in request.iter_content(chunk_size=chunk_size):
                        fp.write(chunk)
            except:
                # Clean up partial download if there was an error
                try:
                    os.unlink(tmp_file)
                except:
                    pass
                raise

            if skip_if_exists and os.path.exists(local_save_file_path):
                # Another parallel process already successfully finished the
                # download - keep the earlier copy.
                os.unlink(tmp_file)
            else:
                os.rename(tmp_file, local_save_file_path)
