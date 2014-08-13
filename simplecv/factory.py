import pkg_resources

from simplecv.base import logger


class FactoryException(Exception):
    pass


class FactoryBase(object):

    def __init__(self):
        super(FactoryBase, self).__init__()
        self.classes = {}

    def register(self, classname, cls=None, module=None):
        """ Register a new classname referring to a real class or
            class definition in a module.
        """
        if classname in self.classes:
            return
        self.classes[classname] = {
            'module': module,
            'cls': cls
        }

    def unregister(self, *classnames):
        """ Unregisters the classnames previously registered via the
            register method. This allows the same classnames to be re-used in
            different contexts.
        """
        for classname in classnames:
            if classname in self.classes:
                self.classes.pop(classname)

    def __getattr__(self, name):
        classes = self.classes
        if name not in classes:
            if name[0] == name[0].lower():
                # if trying to access attributes
                # then raise AttributeError
                raise AttributeError
            raise FactoryException('Unknown class <%s>' % name)

        item = classes[name]
        cls = item['cls']

        # No class to return, import the module
        if cls is None:
            module_name = item['module']
            if module_name:
                module = __import__(name=module_name, fromlist='.')
                if not hasattr(module, name):
                    raise FactoryException(
                        'No class named <%s> in module <%s>' % (
                            name, module_name))
                cls = item['cls'] = getattr(module, name)
            else:
                raise FactoryException('No information to create the class')

        return cls


# Factory instance to use for getting new classes
Factory = FactoryBase()

Factory.register('Image', module='simplecv.image')

Factory.register('Corner', module='simplecv.features.detection')
Factory.register('Line', module='simplecv.features.detection')
Factory.register('Circle', module='simplecv.features.detection')
Factory.register('Barcode', module='simplecv.features.detection')
Factory.register('Chessboard', module='simplecv.features.detection')
Factory.register('Motion', module='simplecv.features.detection')
Factory.register('KeyPoint', module='simplecv.features.detection')
Factory.register('KeypointMatch', module='simplecv.features.detection')
Factory.register('TemplateMatch', module='simplecv.features.detection')
Factory.register('ShapeContextDescriptor',
                 module='simplecv.features.detection')
Factory.register('ROI', module='simplecv.features.detection')

Factory.register('DFT', module='simplecv.dft')


for plugin in pkg_resources.iter_entry_points('simplecv.factory'):
    logger.info('Registering factory class from plugin "{}"'.format(plugin.name))
    Factory.register(plugin.name, module=plugin.module_name)
