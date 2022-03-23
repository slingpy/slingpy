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
from typing import AnyStr


class AppPaths(object):
    """
    Pre-defined application paths.
    """
    def __init__(self, project_root_directory):
        self.project_root_directory = project_root_directory
        """ 
        The project's root directory. All custom python packages must either be located in sub-directory of the root
         directory or must be loaded as a dependency in the current python environment (e.g., in a virtualenv). 
        """

    @staticmethod
    def get_loss_file_path(output_directory: AnyStr) -> AnyStr:
        return os.path.join(output_directory, "losses.pickle")

    @staticmethod
    def get_args_file_path(output_directory: AnyStr) -> AnyStr:
        return os.path.join(output_directory, "args.pickle")

    @staticmethod
    def get_eval_score_dict_path(output_directory: AnyStr) -> AnyStr:
        return os.path.join(output_directory, "eval_score.pickle")

    @staticmethod
    def get_test_score_dict_path(output_directory: AnyStr) -> AnyStr:
        return os.path.join(output_directory, "test_score.pickle")

    @staticmethod
    def get_run_results_path(output_directory: AnyStr) -> AnyStr:
        return os.path.join(output_directory, "run_results.pickle")

    @staticmethod
    def get_model_folder_path(output_directory: AnyStr) -> AnyStr:
        return os.path.join(output_directory, "model")

    @staticmethod
    def get_model_file_path(output_directory: AnyStr, extension: AnyStr) -> AnyStr:
        return os.path.join(output_directory, f"model.{extension}")

    @staticmethod
    def get_prediction_file_path(output_directory: AnyStr, set_name: AnyStr,
                                 extension: AnyStr, output_index: int) -> AnyStr:
        return os.path.join(output_directory, f"{set_name}_predictions.output_{output_index}.{extension}")

    @staticmethod
    def get_thresholded_prediction_file_path(output_directory: AnyStr, set_name: AnyStr,
                                             extension: AnyStr, output_index: int) -> AnyStr:
        return os.path.join(output_directory, f"{set_name}_predictions.output_{output_index}.tresholded.{extension}")
