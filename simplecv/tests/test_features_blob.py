from nose.tools import assert_equals

from simplecv.features.blob import Blob
from simplecv.image import Image
from simplecv.color import Color

import pickle

def test_setstate():
    b = Blob()
    #b_string = pickle.dumps(b)
    #b_new = pickle.loads(b_string)
    b.__setstate__({'m00':50, 'm01':100, 'm02__string':200, 'label__string':'empty'})
    assert_equals(b.m00, 50)
    assert_equals(b.m01, 100)
    assert_equals(b.m02, 200)
    assert_equals(b.label, 'empty')

def test_hull():
    img = Image(source="lenna")
    blobs = img.find_blobs()
    blob = blobs[-1]
    chull = blob.hull()

def test_drawRect():
    img = Image(source="lenna")
    blobs = img.find_blobs()
    blob = blobs[-1]
    blob.draw_rect(color=Color.BLUE, width=-1, alpha=128)
    img.show()

