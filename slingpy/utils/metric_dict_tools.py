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
import numpy as np
from typing import Tuple, Dict, List
from slingpy.utils.logging import info, error


if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class MetricDictTools(object):
    """
    Tools for working with metric dictionaries, i.e. dictionaries in which each entry consists of a named evaluation
    metric and one or more associated metric values.
    """
    @staticmethod
    def aggregate_metric_dicts(eval_scores: List[Dict], test_scores: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Aggregates lists of metric dictionaries into a single dictionary that uses the mean value across all results
        as the representative result. In addition, new entries reflecting the mean ($_results), standard deviation
        ($_std) and the original sub-results themselves ($_results) are added to the aggregated metric dictionary.

        Args:
            eval_scores: A list of evaluation metric dictionaries.
            test_scores: A list of test metric dictionaries.

        Returns:
            Two dictionaries containing the aggregated __eval_scores__ and __test_scores__, respectively.
        """
        output_dicts = ({}, {})
        for dict_list, output_dict in zip([eval_scores, test_scores], output_dicts):
            for key in dict_list[0].keys():
                all_values = []
                for scores in dict_list:
                    all_values.append(scores[key])
                if not key.endswith("_results"):
                    output_dict[key] = np.mean(all_values)
                output_dict[key + "_std"] = np.std(all_values)
                output_dict[key + "_results"] = all_values
        return output_dicts

    @staticmethod
    def save_metric_dict(metric_dict, file_path):
        """
        Saves a metric dictionary to disk using pickle.

        Args:
            metric_dict: The metric dictionary.
            file_path: The file path the metric dictionary should be written to.
        """
        with open(file_path, "wb") as fp:
            pickle.dump(metric_dict, fp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_metric_dict(file_path):
        """
        Loads a metric dictionary from disk that was saved using __save_metric_dict__.

        Args:
            file_path: The file path the metric dictionary should be loaded from.

        Returns:
            The loaded metric dictionary.
        """
        with open(file_path, "rb") as fp:
            score_dict = pickle.load(fp)
            return score_dict

    @staticmethod
    def print_metric_statistics(metric_dicts, name):
        """
        Prints metric statistics of a list of metric dictionaries.

        Args:
            metric_dicts: A list of metric dictionaries to print.
            name: The name of the metric dictionaries printout.
        """
        info("{} results (N={:d}) are:".format(name, len(metric_dicts)))
        for key in metric_dicts[0].keys():
            try:
                values = list(map(lambda x: x[key], metric_dicts))
                info(key, "=", np.mean(values), "+-", np.std(values),
                     "CI=(", np.percentile(values, 2.5), ",", np.percentile(values, 97.5), "),",
                     "median=", np.median(values),
                     "min=", np.min(values),
                     "max=", np.max(values))
            except:
                error("Could not get key", key, "for all metric dicts.")
