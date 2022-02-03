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
import six
import json
import shutil
import tarfile
import tempfile
from typing import AnyStr, Type
from abc import ABCMeta, abstractmethod
from slingpy.models.abstract_base_model import AbstractBaseModel


@six.add_metaclass(ABCMeta)
class TarfileSerialisationBaseModel(AbstractBaseModel):
    def get_config(self, deep=True):
        config = self.get_params(deep=deep)
        return config

    @staticmethod
    def get_config_file_name():
        return "model_config.json"

    @staticmethod
    def load_config(save_folder_path):
        config_file_name = TarfileSerialisationBaseModel.get_config_file_name()
        config_file_path = os.path.join(save_folder_path, config_file_name)
        with open(config_file_path, "r") as fp:
            config = json.load(fp)
        return config

    @classmethod
    def load(cls: Type[AbstractBaseModel], file_path: AnyStr) -> AbstractBaseModel:
        temp_dir = tempfile.mkdtemp()
        with tarfile.open(file_path) as tar:
            tar.extractall(path=temp_dir)
        tar_root_dir = os.path.join(temp_dir, TarfileSerialisationBaseModel.get_root_tar_folder_name())
        instance = cls.load_folder(tar_root_dir)
        shutil.rmtree(temp_dir)
        return instance

    @staticmethod
    def get_save_file_extension() -> AnyStr:
        return "tar.gz"

    @staticmethod
    def get_root_tar_folder_name() -> AnyStr:
        return "model"

    def save(self, file_path: AnyStr):
        temp_dir = tempfile.mkdtemp()
        self.save_folder(temp_dir)
        with tarfile.open(file_path, "w:gz") as tar:
            tar.add(temp_dir, arcname=TarfileSerialisationBaseModel.get_root_tar_folder_name())
        shutil.rmtree(temp_dir)

    def save_config(self, directory_path, config, config_file_name, overwrite, outer_class):
        already_exists_exception_message = "__directory_path__ already contains a saved" + outer_class.__name__ + \
                                           " instance and __overwrite__ was set to __False__. Conflicting file: {}"
        config_file_path = os.path.join(directory_path, config_file_name)
        if os.path.exists(config_file_path) and not overwrite:
            raise ValueError(already_exists_exception_message.format(config_file_path))
        else:
            with open(config_file_path, "w") as fp:
                json.dump(config, fp)

    @abstractmethod
    def save_folder(self, save_folder_path, overwrite=True):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load_folder(cls: Type["TarfileSerialisationBaseModel"], save_folder_path: AnyStr) -> AbstractBaseModel:
        raise NotImplementedError()
