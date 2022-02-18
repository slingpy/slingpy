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
from pathlib import Path
from slingpy.utils.logging import warn
from typing import AnyStr, Dict, List, Callable


class PluginTools(object):
    """ Tools for working with python plugins loaded at runtime using __importlib__. """
    @staticmethod
    def get_available_plugins(search_directory: AnyStr, check_for_subclasses: List = None,
                              module_name_prefix: AnyStr = "",
                              new_module_encountered_hook: Callable = lambda module_name, module: None):
        classes = []
        for path in Path(search_directory).rglob('*.py'):
            modname = module_name_prefix + os.path.basename(path)[:-3]
            spec = importlib.util.spec_from_file_location(modname, path)
            module = importlib.util.module_from_spec(spec)
            new_module_encountered_hook(modname, module)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and all([issubclass(obj, subclass) for subclass in check_for_subclasses]):
                    classes.append(obj)
        return classes

    @staticmethod
    def load_plugin(name: AnyStr, search_directory: AnyStr,
                    module_name_prefix: AnyStr = "",
                    new_module_encountered_hook: Callable = lambda module_name, module: None):
        """
        Loads a plugin dynamically.

        Args:
            name: The name of the plugin to be loaded. Must be exact match with the class name given in the code.
            search_directory: The directory to search for the plugin.
            module_name_prefix: Prefix to apply to dynamically loaded module's names.
            new_module_encountered_hook: Callable hook to trigger if a new module is loaded dynamically.

        Returns:
            The loaded plugin class.
        """
        for module_path in glob.glob(search_directory + "/*.py"):
            modname = module_name_prefix + os.path.basename(module_path)[:-3]
            spec = importlib.util.spec_from_file_location(modname, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            new_module_encountered_hook(modname, module)
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
    def instantiate_plugin(name: AnyStr, search_directory: AnyStr, kwargs: Dict = None,
                           module_name_prefix: AnyStr = "",
                           new_module_encountered_hook: Callable = lambda module_name, module: None):
        """
        Instantiates a plugin instance dynamically.

        Args:
            name: The name of the plugin. Must be exact match with the class name given in the code.
            search_directory: The directory to search for the plugin.
            kwargs: The instance arguments to pass on instantiation.
            module_name_prefix: Prefix to apply to dynamically loaded module's names.
            new_module_encountered_hook: Callable hook to trigger if a new module is loaded dynamically.

        Returns:
            The loaded plugin instance.
        """
        if kwargs is None:
            kwargs = {}

        cls = PluginTools.load_plugin(name=name, search_directory=search_directory,
                                      module_name_prefix=module_name_prefix,
                                      new_module_encountered_hook=new_module_encountered_hook)
        if cls is not None:
            available_model_params = PluginTools.get_available_instance_parameters(cls, kwargs)
            instance = cls(**available_model_params)
            return instance
        else:
            return None
