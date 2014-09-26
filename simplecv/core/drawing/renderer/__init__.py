from simplecv.core.pluginsystem import plugin_dict


@plugin_dict('renderers')
class Renderer(object):

    renderers = {}

    @staticmethod
    def render(layer, image, renderer):
        renderer = Renderer.renderers[renderer]()
        return renderer.render(layer, image)
