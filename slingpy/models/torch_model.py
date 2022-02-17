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
import torch
import shutil
import tempfile
import numpy as np
from time import time
import torch.nn as nn
import torch.optim as optim
from slingpy.utils.logging import info
from torch.utils.data import DataLoader
from slingpy.losses.torch_loss import TorchLoss
from slingpy.utils.plugin_tools import PluginTools
from typing import Optional, Type, AnyStr, Callable, List
from slingpy.models.abstract_base_model import AbstractBaseModel
from slingpy.transforms.abstract_transform import AbstractTransform
from slingpy.data_access.data_sources.abstract_data_source import AbstractDataSource
from slingpy.data_access.data_sources.composite_data_source import CompositeDataSource
from slingpy.models.tarfile_serialisation_base_model import TarfileSerialisationBaseModel


if sys.version_info < (3, 0, 0):
    import cPickle as pickle
else:
    import pickle


class TorchModel(TarfileSerialisationBaseModel):
    def __init__(self,
                 base_module: nn.Module,  # Note: __base_module__ must also implement ArgumentDictionary.
                 loss: TorchLoss,
                 preprocess_x_fn: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None,
                 preprocess_y_fn: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None,
                 postprocess_y_fn: Optional[Callable[[List[torch.Tensor]], List[torch.Tensor]]] = None,
                 collate_fn: Optional[Callable] = None,
                 target_transformer: Optional[AbstractTransform] = None,
                 learning_rate: float = 1e-3,
                 early_stopping_patience: int = 13,
                 batch_size: int = 256,
                 num_epochs: int = 100,
                 l2_weight: float = 1e-4):
        super(TorchModel, self).__init__()
        self.base_module = base_module
        self.target_transformer = target_transformer
        self.loss = loss
        self.preprocess_x_fn = preprocess_x_fn
        self.preprocess_y_fn = preprocess_y_fn
        self.postprocess_y_fn = postprocess_y_fn
        self.collate_fn = collate_fn
        self.model = None
        self.l2_weight = l2_weight
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

    def get_config(self, deep=True):
        config = super(TorchModel, self).get_config(deep=deep)
        del config["base_module"]
        del config["target_transformer"]
        del config["loss"]
        del config["preprocess_x_fn"]
        del config["preprocess_y_fn"]
        del config["postprocess_y_fn"]
        return config

    def get_model_prediction(self, data: List[torch.Tensor]) -> List[torch.Tensor]:
        y_pred = self.model(data)
        if self.postprocess_y_fn:
            y_pred = self.postprocess_y_fn(y_pred)
        if self.target_transformer:
            y_pred = self.target_transformer.inverse_transform(y_pred)
        return y_pred

    def predict(self, dataset_x: AbstractDataSource, batch_size: int = 256,
                row_names: List[AnyStr] = None) -> List[np.ndarray]:
        if self.model is None:
            self.model = self.build()
        if row_names is None:
            row_names = dataset_x.get_row_names()

        self.model.eval()
        all_ids, y_preds, y_trues = [], [], []
        while len(row_names) > 0:
            current_indices = row_names[:batch_size]
            row_names = row_names[batch_size:]
            data = dataset_x.get_data(current_indices)
            data = list(map(torch.from_numpy, data))
            y_pred = self.get_model_prediction(data)
            y_preds.append(y_pred)
        y_preds = list(map(lambda y_preds_i: torch.cat(y_preds_i, dim=0).detach().numpy(), zip(*y_preds)))
        return y_preds

    def _make_loader(self, data_source, is_training=False):
        loader = DataLoader(
            data_source,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            pin_memory=False,
            drop_last=False,  # Must be set for training in case batch_size == 1 for last batch.
            timeout=0,
            worker_init_fn=None,
        )
        return loader

    def fit(self, train_x: AbstractDataSource, train_y: Optional[AbstractDataSource] = None,
            validation_set_x: Optional[AbstractDataSource] = None,
            validation_set_y: Optional[AbstractDataSource] = None) -> AbstractBaseModel:
        if self.model is None:
            self.model = self.build()

        temp_dir = tempfile.mkdtemp()
        model_file_path = os.path.join(temp_dir, "model.pt")

        # Save once up front in case training does not converge.
        torch.save(self.model.state_dict(), model_file_path)
        training_set = CompositeDataSource([train_x, train_y])
        loader_train = self._make_loader(training_set.to_torch(), is_training=True)
        loader_val = None
        if validation_set_y and validation_set_x:
            validation_set = CompositeDataSource([validation_set_x, validation_set_y])
            loader_val = self._make_loader(validation_set.to_torch())

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_weight)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)

        def get_output_and_loss(model_inputs, model_labels):
            if self.preprocess_x_fn is not None:
                model_inputs = self.preprocess_x_fn(model_inputs)

            if self.preprocess_y_fn is not None:
                model_labels = self.preprocess_y_fn(model_labels)

            y_pred_i = [y.cpu() for y in self.model([m.to(device) for m in model_inputs])]
            loss = self.loss(y_pred_i, model_labels)
            return y_pred_i, loss

        best_val_loss, num_epochs_no_improvement, num_labels = float("inf"), 0, len(train_y.get_shape())
        for epoch in range(self.num_epochs):  # Loop over the dataset multiple times.
            self.model.train()
            start_time = time()

            train_loss, num_batches_seen = 0.0, 0
            for i, batch_data in enumerate(loader_train):
                inputs, labels = batch_data[:-num_labels], batch_data[-num_labels:]

                if self.target_transformer is not None:
                    labels = self.target_transformer.transform(labels)

                optimizer.zero_grad()
                y_pred_i, loss = get_output_and_loss(inputs, labels)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches_seen += 1
            train_loss /= num_batches_seen

            val_loss = train_loss
            if loader_val:
                self.model.eval()
                val_loss, num_batches_seen_val = 0.0, 0
                for batch_data in loader_val:
                    inputs, labels = batch_data[:-num_labels], batch_data[-num_labels:]

                    if self.target_transformer is not None:
                        labels = self.target_transformer.transform(labels)

                    y_pred_i, loss = get_output_and_loss(inputs, labels)

                    val_loss += loss
                    num_batches_seen_val += 1
                val_loss /= num_batches_seen_val

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_file_path)
                num_epochs_no_improvement = 0
            else:
                num_epochs_no_improvement += 1

            epoch_duration = time() - start_time
            info(f"Epoch {epoch+1:d}/{self.num_epochs:d} [{epoch_duration:.2f}s]: "
                 f"loss = {train_loss:.4f}, val_loss = {val_loss:.4f}")

            if num_epochs_no_improvement >= self.early_stopping_patience:
                break

        info("Resetting to best encountered model at", model_file_path, ".")

        # Reset to the best model observed in training.
        self.model = self.model.cpu()
        self.model.load_state_dict(torch.load(model_file_path))
        shutil.rmtree(temp_dir)  # Cleanup temporary directory after training.
        return self

    def build(self) -> nn.Module:
        available_model_params = PluginTools.get_available_instance_parameters(
            self.base_module, self.base_module.get_params()
        )
        return self.base_module.__class__(**available_model_params)

    @classmethod
    def get_model_save_file_name(cls) -> AnyStr:
        return "model.pt"

    @classmethod
    def get_target_transformer_save_file_name(cls) -> AnyStr:
        return "target_transformer.pickle"

    @classmethod
    def get_base_module_save_file_name(cls) -> AnyStr:
        return "base_module.pickle"

    @classmethod
    def get_loss_save_file_name(cls) -> AnyStr:
        return "loss.pickle"

    @classmethod
    def get_preprocess_x_fn_save_file_name(cls) -> AnyStr:
        return "preprocess_x_fn.pickle"

    @classmethod
    def get_preprocess_y_fn_save_file_name(cls) -> AnyStr:
        return "preprocess_y_fn.pickle"

    @classmethod
    def get_postprocess_y_fn_save_file_name(cls) -> AnyStr:
        return "postprocess_y_fn.pickle"

    @classmethod
    def get_collate_fn_save_file_name(cls) -> AnyStr:
        return "collate_fn.pickle"

    def save_folder(self, save_folder_path, overwrite=True):
        self.save_config(
            save_folder_path,
            self.get_config(deep=False),
            self.get_config_file_name(),
            overwrite,
            self.__class__
        )
        model_save_path = os.path.join(save_folder_path, self.get_model_save_file_name())
        torch.save(
            self.model.state_dict(), model_save_path
        )

        base_module_path = os.path.join(save_folder_path, self.get_base_module_save_file_name())
        with open(base_module_path, "wb") as save_file:
            pickle.dump(self.base_module, save_file, pickle.HIGHEST_PROTOCOL)

        loss_save_path = os.path.join(save_folder_path, self.get_loss_save_file_name())
        with open(loss_save_path, "wb") as save_file:
            pickle.dump(self.loss, save_file, pickle.HIGHEST_PROTOCOL)

        if self.target_transformer is not None:
            target_transformer_save_path = os.path.join(save_folder_path, self.get_target_transformer_save_file_name())
            with open(target_transformer_save_path, "wb") as save_file:
                pickle.dump(self.target_transformer, save_file, pickle.HIGHEST_PROTOCOL)

        if self.preprocess_x_fn is not None:
            preprocess_x_fn_path = os.path.join(save_folder_path, self.get_preprocess_x_fn_save_file_name())
            with open(preprocess_x_fn_path, "wb") as save_file:
                pickle.dump(self.preprocess_x_fn, save_file, pickle.HIGHEST_PROTOCOL)

        if self.preprocess_y_fn is not None:
            preprocess_y_fn_path = os.path.join(save_folder_path, self.get_preprocess_y_fn_save_file_name())
            with open(preprocess_y_fn_path, "wb") as save_file:
                pickle.dump(self.preprocess_y_fn, save_file, pickle.HIGHEST_PROTOCOL)

        if self.collate_fn is not None:
            collate_fn_path = os.path.join(save_folder_path, self.get_collate_fn_save_file_name())
            with open(collate_fn_path, "wb") as save_file:
                pickle.dump(self.collate_fn, save_file, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_folder(cls: Type[TarfileSerialisationBaseModel], save_folder_path: AnyStr) -> AbstractBaseModel:
        config = cls.load_config(save_folder_path)

        base_module_path = os.path.join(save_folder_path, cls.get_base_module_save_file_name())
        with open(base_module_path, "rb") as load_file:
            base_module = pickle.load(load_file)

        config["base_module"] = base_module
        config["loss"] = None

        instance = cls(**config)
        weight_list = torch.load(os.path.join(save_folder_path, cls.get_model_save_file_name()))
        instance.model = instance.build()
        instance.model.load_state_dict(weight_list)

        loss_save_path = os.path.join(save_folder_path, cls.get_loss_save_file_name())
        with open(loss_save_path, "rb") as load_file:
            instance.loss = pickle.load(load_file)

        target_transformer_save_path = os.path.join(save_folder_path, cls.get_target_transformer_save_file_name())
        if os.path.isfile(target_transformer_save_path):
            with open(target_transformer_save_path, "rb") as load_file:
                instance.target_transformer = pickle.load(load_file)
        else:
            instance.target_transformer = None

        preprocess_x_fn_path = os.path.join(save_folder_path, cls.get_preprocess_x_fn_save_file_name())
        if os.path.isfile(preprocess_x_fn_path):
            with open(preprocess_x_fn_path, "rb") as load_file:
                instance.preprocess_x_fn = pickle.load(load_file)
        else:
            instance.preprocess_x_fn = None

        preprocess_y_fn_path = os.path.join(save_folder_path, cls.get_preprocess_y_fn_save_file_name())
        if os.path.isfile(preprocess_y_fn_path):
            with open(preprocess_y_fn_path, "rb") as load_file:
                instance.preprocess_y_fn = pickle.load(load_file)
        else:
            instance.preprocess_y_fn = None

        postprocess_y_fn_path = os.path.join(save_folder_path, cls.get_postprocess_y_fn_save_file_name())
        if os.path.isfile(postprocess_y_fn_path):
            with open(postprocess_y_fn_path, "rb") as load_file:
                instance.postprocess_y_fn = pickle.load(load_file)
        else:
            instance.postprocess_y_fn = None

        collate_fn_path = os.path.join(save_folder_path, cls.get_collate_fn_save_file_name())
        if os.path.isfile(collate_fn_path):
            with open(collate_fn_path, "rb") as load_file:
                instance.collate_fn = pickle.load(load_file)
        else:
            instance.collate_fn = None
        return instance
