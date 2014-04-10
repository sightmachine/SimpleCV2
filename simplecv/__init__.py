__version__ = '2.0.0'

# from simplecv.base import *
# from simplecv.camera import *
# from simplecv.color import *
# from simplecv.display import *
# from simplecv.features import *
# from simplecv.image_class import *
# from simplecv.stream import *
# from simplecv.font import *
# from simplecv.color_model import *
# from simplecv.drawing_layer import *
# from simplecv.segmentation import *
# from simplecv.machine_learning import *
# from simplecv.linescan import *
# from simplecv.dft import DFT

import os
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


if __name__ == '__main__':
    import sys
    from simplecv.shell import shell
    shell.main(sys.argv)
