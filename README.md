# slingpy

![Python version](https://img.shields.io/badge/Python-3.8-blue)
![Library version](https://img.shields.io/badge/Version-0.2.2-blue)

The `slingpy` python package provides starter code for structured, reproducible and maintainable machine learning
projects. `slingpy` aims to be maximally extensible while maintaining sensible defaults. It is agnostic in terms of modelling backend
(e.g., supporting `scikit-learn`, `torch` and `tensorflow`) and suitable for both production and research-grade machine learning projects.

`slingpy` contains utilities for standard model evaluation workflows, such as nested cross validation, model serialisation, dataset handling, managing high performance computing (HPC) interfaces such as slurm, and by 
default writes all experiment artefacts to disk.

## Install

```bash
pip install slingpy
```

## Use

A minimal `slingpy` project consists of a base application that defines your basic modelling workflow.

```python
import slingpy as sp
from typing import AnyStr, Dict, List
from sklearn.linear_model import LogisticRegression


class MyApplication(sp.AbstractBaseApplication):
    def __init__(self, output_directory: AnyStr = "",
                 schedule_on_slurm: bool = False,
                 split_index_outer: int = 0,
                 split_index_inner: int = 0,
                 num_splits_outer: int = 5,
                 num_splits_inner: int = 5):
        super(MyApplication, self).__init__(
            output_directory=output_directory,
            schedule_on_slurm=schedule_on_slurm,
            split_index_outer=split_index_outer,
            split_index_inner=split_index_inner,
            num_splits_outer=num_splits_outer,
            num_splits_inner=num_splits_inner
        )

    def get_metrics(self, set_name: AnyStr) -> List[sp.AbstractMetric]:
        return [
            sp.metrics.AreaUnderTheCurve()
        ]

    def load_data(self) -> Dict[AnyStr, sp.AbstractDataSource]:
        data_source_x, data_source_y = sp.datasets.Iris.load_data(self.output_directory)

        stratifier = sp.StratifiedSplit()
        rest_indices, training_indices = stratifier.split(data_source_y, test_set_fraction=0.6,
                                                          split_index=self.split_index_inner)
        validation_indices, test_indices = stratifier.split(data_source_y.subset(rest_indices), test_set_fraction=0.5,
                                                            split_index=self.split_index_outer)

        return {
            "training_set_x": data_source_x.subset(training_indices),
            "training_set_y": data_source_y.subset(training_indices),
            "validation_set_x": data_source_x.subset(validation_indices),
            "validation_set_y": data_source_y.subset(validation_indices),
            "test_set_x": data_source_x.subset(test_indices),
            "test_set_y": data_source_y.subset(test_indices)
        }

    def train_model(self) -> sp.AbstractBaseModel:
        model = sp.SklearnModel(LogisticRegression())
        model.fit(self.datasets.training_set_x, self.datasets.training_set_y)
        return model


if __name__ == "__main__":
    app = sp.instantiate_from_command_line(MyApplication)
    app._run()
```

Your new app can then be executed locally from the command line using:
```bash
python /project_path/my_application.py
```

`slingpy` also enables execution of your project on a remote HPC cluster, e.g. via slurm, by using:
```bash
python /project_path/my_application.py --do_schedule_on_slurm
```

Application parameters are automatically parsed from the command line, e.g.:
```bash
python /project_path/my_application.py --output_directory=/path/to/output/dir
```

### License

[License](LICENSE.txt)

### Authors

Patrick Schwab, GlaxoSmithKline plc<br/>
Arash Mehrjou, GlaxoSmithKline plc<br/>
Andrew Jesson, University of Oxford<br/>
Ashkan Soleymani, MIT

## Acknowledgements

PS and AM are employees and shareholders of GlaxoSmithKline plc.