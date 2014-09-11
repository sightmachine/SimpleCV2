from simplecv.factory import Factory


def test_factory():
    img = Factory.Image(source='simplecv')
    assert img is not None
    assert Factory.Corner(i=img, at_x=5, at_y=5)
    assert Factory.Line(i=img, line=((5, 5), (10, 10)))
