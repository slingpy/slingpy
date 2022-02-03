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
import glob
import inspect
import importlib
import importlib.util
from typing import AnyStr, Dict
from slingpy.utils.logging import warn


class PluginTools(object):
    """ Tools for working with python plugins loaded at runtime using __importlib__. """
    @staticmethod
    def load_plugin(name: AnyStr, search_directory: AnyStr):
        """
        Loads a plugin dynamically.

        Args:
            name: The name of the plugin to be loaded. Must be exact match with the class name given in the code.
            search_directory: The directory to search for the plugin.

        Returns:
            The loaded plugin class.
        """
        for module_path in glob.glob(search_directory + "/*.py"):
            modname = os.path.basename(module_path)[:-3]
            spec = importlib.util.spec_from_file_location(modname, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls = getattr(module, name, None)
            if cls is not None:
                return cls
        warn(f"No model type found for name {name}. Returning None.")
        return None

    @staticmethod
    def get_available_instance_parameters(cls, kwargs):
        parameters = inspect.signature(cls.__init__).parameters
        available_model_params = {
            k: kwargs[k] if k in kwargs else parameters[k].default for k in parameters.keys()
            if k in kwargs or parameters[k].default is not parameters[k].empty
        }
        return available_model_params

    @staticmethod
    def instantiate_plugin(name: AnyStr, search_directory: AnyStr, kwargs: Dict = None):
        """
        Instantiates a plugin instance dynamically.

        Args:
            name: The name of the plugin. Must be exact match with the class name given in the code.
            search_directory: The directory to search for the plugin.
            kwargs: The instance arguments to pass on instantiation.

        Returns:
            The loaded plugin instance.
        """
        if kwargs is None:
            kwargs = {}

        cls = PluginTools.load_plugin(name=name, search_directory=search_directory)
        if cls is not None:
            available_model_params = PluginTools.get_available_instance_parameters(cls, kwargs)
            instance = cls(**available_model_params)
            return instance
        else:
            return None
