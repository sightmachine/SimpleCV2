import os
import tempfile

from simplecv.color_model import ColorModel
from simplecv.image import Image
from simplecv.tests.utils import perform_diff

testimage = "../data/sampleimages/9dots4lines.png"


def test_color_colormap_build():
    cm = ColorModel()
    cm.add((127, 127, 127))
    assert cm.contains((127, 127, 127))
    cm.remove((127, 127, 127))

    cm.remove((0, 0, 0))
    cm.remove((255, 255, 255))
    cm.add((0, 0, 0))
    cm.add([(0, 0, 0), (255, 255, 255)])
    cm.add([(255, 0, 0), (0, 255, 0)])
    img = cm.threshold(Image(source=testimage))

    tmp_dir = tempfile.gettempdir()
    tmp_txt = os.path.join(tmp_dir, 'temp.txt')
    cm.save(tmp_txt)

    cm2 = ColorModel()
    cm2.load(tmp_txt)
    img = Image(source="logo")
    img2 = cm2.threshold(img)
    cm2.add((0, 0, 255))
    img3 = cm2.threshold(img)
    cm2.add((255, 255, 0))
    cm2.add((0, 255, 255))
    cm2.add((255, 0, 255))
    img4 = cm2.threshold(img)
    cm2.add(img)
    img5 = cm2.threshold(img)

    results = [img, img2, img3, img4, img5]
    name_stem = "test_color_colormap_build"
    perform_diff(results, name_stem)
