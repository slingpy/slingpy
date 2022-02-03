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
import numpy as np
from collections import Counter
from typing import Tuple, List, AnyStr
from slingpy.utils.to_categorical import to_categorical
from slingpy.utils.iterative_stratification import IterativeStratification
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource


class StratifiedSplit(object):
    @staticmethod
    def make_synthetic_labels_for_stratification(num_bins=5, max_num_unique_values=10,
                                                 label_candidates=list([])):
        synthetic_labels = []
        for arg_idx in range(len(label_candidates)):
            arg = label_candidates[arg_idx]
            if len(arg) == 0:
                raise ValueError("Length of synthetic label inputs should not be zero.")

            if len(set(arg)) <= max_num_unique_values:
                if len(set(arg)) == 2:
                    more_labels = np.array(arg).astype(int)
                else:
                    more_labels = to_categorical(np.array(arg).astype(int), num_classes=len(set(arg)))
            else:
                arg = arg.astype(float)
                nan_indices = list(map(lambda xi: np.isnan(xi), arg))

                if np.alltrue(nan_indices):
                    continue

                assignments = np.digitize(arg, np.linspace(np.min(arg[np.logical_not(nan_indices)]),
                                                           np.max(arg[np.logical_not(nan_indices)]), num_bins)) - 1
                more_labels = to_categorical(assignments,
                                             num_classes=num_bins)

            synthetic_labels.append(more_labels)
        synthetic_labels = np.column_stack(synthetic_labels)
        return synthetic_labels

    def merge_groups_smaller_than(self, converted_synthetic_labels, min_group_size: int = 5):
        counts = Counter(converted_synthetic_labels)
        sorted_keys = sorted(range(len(counts)), key=lambda key: counts[key])
        for k in range(len(sorted_keys)):
            key = sorted_keys[k]
            if counts[key] < min_group_size:
                next_idx = (k + 1) % len(sorted_keys)
                converted_synthetic_labels[converted_synthetic_labels == key] = sorted_keys[next_idx]
                counts[sorted_keys[next_idx]] += counts[key]
                counts[key] = 0
        return converted_synthetic_labels

    def split(self, dataset: AbstractDataSource, test_set_fraction: float = 0.2,
              min_group_size: int = 5, seed: int = 0, num_splits: int = 2,
              split_index: int = 0, max_num_unique_values: int = 10,
              bin_size: int = 5) -> Tuple[List[AnyStr], List[AnyStr]]:
        split_index = max(0, split_index)  # __split_index__ must be at least 0.
        data = np.concatenate(dataset.get_data(), axis=-1)
        split_ids = dataset.get_row_names()

        synthetic_labels = []
        for j in range(data.shape[-1]):
            unique_values = sorted(list(set(data[:, j])))
            if len(unique_values) < max_num_unique_values:
                index_map = dict(zip(unique_values, range(len(unique_values))))
                labels = list(map(lambda xi: index_map[xi], data[:, j]))
            else:
                labels = data[:, j]
            synthetic_labels.append(labels)

        x = np.arange(len(data))
        converted_synthetic_labels = StratifiedSplit.make_synthetic_labels_for_stratification(
            label_candidates=synthetic_labels,
            num_bins=bin_size,
            max_num_unique_values=max_num_unique_values
        )
        converted_synthetic_labels = np.squeeze(
            converted_synthetic_labels.astype(int).dot(2 ** np.arange(converted_synthetic_labels.shape[-1])[::-1])
        )
        synthetic_labels_map = dict(zip(sorted(list(set(converted_synthetic_labels))),
                                        range(len(set(converted_synthetic_labels)))))
        converted_synthetic_labels = np.array(list(
            map(lambda xi: synthetic_labels_map[xi], converted_synthetic_labels)
        ))
        converted_synthetic_labels = self.merge_groups_smaller_than(converted_synthetic_labels,
                                                                    min_group_size=min_group_size)

        if num_splits == 2:
            sss = IterativeStratification(n_splits=2, order=2, random_state=seed,
                                          sample_distribution_per_fold=[test_set_fraction, 1.0 - test_set_fraction])
        else:
            sss = IterativeStratification(n_splits=num_splits, order=2, random_state=seed)

        for _ in range(split_index + 1):
            train_index, test_index = next(sss.split(x, converted_synthetic_labels[:, np.newaxis]))

        assert len(set(train_index).intersection(set(test_index))) == 0
        train_indices, test_indices = np.array(split_ids)[train_index], np.array(split_ids)[test_index]
        return train_indices, test_indices
