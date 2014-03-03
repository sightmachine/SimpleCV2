__version__ = '1.3.0'

from simplecv.base import *
from simplecv.camera import *
from simplecv.color import *
from simplecv.display import *
from simplecv.features import *
from simplecv.image_class import *
from simplecv.stream import *
from simplecv.font import *
from simplecv.color_model import *
from simplecv.drawing_layer import *
from simplecv.segmentation import *
from simplecv.machine_learning import *
from simplecv.linescan import *
from simplecv.dft import DFT

if (__name__ == '__main__'):
    from simplecv.shell import *
    main(sys.argv)
