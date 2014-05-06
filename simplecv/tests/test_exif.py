from nose.tools import assert_equals, assert_almost_equals
from simplecv import exif

# from simplecv.image import Image

def test_exif_make_string():
    seq = ['\x1b', '\t', '\n']  
    result = exif.make_string(seq)
    assert_equals(result, seq)

# def test_exif_make_string_uc():

def test_exif_nikon_ev_bias():
    seq = [256, 2, 3]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "")

    seq = [252, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "-2/3 EV")

    seq = [253, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "-1/2 EV")
    
    seq = [254, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "-1/3 EV")

    seq = [0, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "0 EV")

    seq = [2, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "+1/3 EV")

    seq = [3, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "+1/2 EV")

    seq = [4, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "+2/3 EV")

    seq = [0, 1, 5, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "0 EV")

    seq = [120, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "+20 EV")

    seq = [136, 1, 6, 0]
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "-20 EV")

    seq = [3, 1, 5, 0]    
    r = exif.nikon_ev_bias(seq)
    assert_equals(r, "+3/5 EV")

def test_exif_olympus_special_mode():
    v = [4, 1, 2]
    r = exif.olympus_special_mode(v)
    assert_equals(r, [4, 1, 2])

    v = [3, 1, 1]
    r = exif.olympus_special_mode(v)
    assert_equals(r, 'Panorama - sequence 1 - Left to right')

def test_exif_s2n_intel():
    s = 'A'
    r = exif.s2n_intel(s)
    assert_equals(r, 65L)

def test_exif_IfdTag_str():
    t = exif.IfdTag("Image Width", 256, 3, (15, 20), 0, 2)
    retval = t.__str__()
    assert_equals(retval, "Image Width")

def test_exif_IfdTag_repr():
    t = exif.IfdTag("Image Width", 256, 3, (15, 20), 0, 2)
    retval = t.__repr__()
    assert_equals(retval, "(0x0100) Short=Image Width @ 0")

def test_exif_ExifHeader_s2n():
    f = open("../data/test/standard/test_exif_Exifheader0.txt")
    h = exif.ExifHeader(f, 'I', 0, False, False)
    val = h.s2n(0, 1)
    assert_equals(val, 65L)
    
    h = exif.ExifHeader(f, 'M', 0, False, False)    
    val = h.s2n(0, 1)
    assert_equals(val, 65)

    f = open("../data/test/standard/test_exif_Exifheader1.txt")
    h = exif.ExifHeader(f, 'I', 0, False, False)
    val = h.s2n(0, 1, 1)
    assert_equals(val, -30L)

def test_exif_ExifHeader_n2s():
    f = open("../data/test/standard/test_exif_Exifheader0.txt")
    h = exif.ExifHeader(f, 'I', 0, False, False)
    val = h.n2s(0, 1)
    assert_equals(val, '\x00')

    h = exif.ExifHeader(f, 'M', 0, False, False)    
    val = h.n2s(0, 1)
    assert_equals(val, '\x00')

# def test_exif_ExifHeader_dump_ifd():
