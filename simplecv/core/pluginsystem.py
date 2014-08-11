import functools
import types

import pkg_resources

from simplecv.base import logger, convert_camel_case_to_underscore


def plugin_method(for_class, static=False, func=None):
    """ Decorator for methods declared outside of the class
    """
    if not isinstance(for_class, str):
        raise TypeError('First argument must be a class name.')

    if func is None:
        return functools.partial(plugin_method, for_class, static)

    if static:
        func._static_plugin_method = for_class
    else:
        func._plugin_method = for_class

    return func


def apply_plugins(cls):
    """ Decorator for the class that can have methods declared outside of the class
    """
    enty_point_group = 'simplecv.' + convert_camel_case_to_underscore(cls.__name__)
    load_plugins(cls, enty_point_group)
    return cls


def load_plugins(cls_inst, enty_point_group):
    for plugin in pkg_resources.iter_entry_points(enty_point_group):
        logger.info('Loading plugin "{}" for {}'.format(plugin.name, cls_inst.__name__))

        try:
            mod = plugin.load()
        except Exception:
            logger.exception('Unable to load plugin "{}"'.format(plugin.name))
            continue

        if isinstance(mod, types.ModuleType):
            for mod_item_name in dir(mod):
                mod_item = getattr(mod, mod_item_name)
                if isinstance(mod_item, types.FunctionType):
                    if hasattr(mod_item, '_plugin_method') \
                            and mod_item._plugin_method == cls_inst.__name__:
                        setattr(cls_inst, mod_item.__name__, mod_item)
                    elif hasattr(mod_item, '_static_plugin_method') \
                            and mod_item._static_plugin_method == cls_inst.__name__:
                        setattr(cls_inst, mod_item.__name__, staticmethod(mod_item))

        elif isinstance(mod, (types.FunctionType, types.MethodType)):
            setattr(cls_inst, mod.__name__, mod)
