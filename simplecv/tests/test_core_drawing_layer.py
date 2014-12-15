from nose.tools import assert_list_equal, assert_false, assert_true
from simplecv.color import Color

from simplecv.core.drawing.layer import DrawingLayer


def test_init():
    dl = DrawingLayer()
    assert_list_equal(
        dl,
        [('set_default_alpha', (255,), {}),
         ('set_default_color', (Color.BLACK,), {})]
    )

    dl.line((10, 10), (20, 20), color=Color.BLUE)

    assert_list_equal(
        dl,
        [('set_default_alpha', (255,), {}),
         ('set_default_color', (Color.BLACK,), {}),
         ('line', ((10, 10), (20, 20)), {'color': Color.BLUE})]
    )


def test_add():
    dl1 = DrawingLayer()
    dl2 = DrawingLayer()
    assert_list_equal(
        dl1 + dl2,
        [('set_default_alpha', (255,), {}),
         ('set_default_color', (Color.BLACK,), {}),
         ('set_default_alpha', (255,), {}),
         ('set_default_color', (Color.BLACK,), {})]
    )

    dl1.line((10, 10), (20, 20), color=Color.BLUE)
    dl2.circle((10, 10), 15)

    assert_list_equal(
        dl1 + dl2,
        [('set_default_alpha', (255,), {}),
         ('set_default_color', (Color.BLACK,), {}),
         ('line', ((10, 10), (20, 20)), {'color': Color.BLUE}),
         ('set_default_alpha', (255,), {}),
         ('set_default_color', (Color.BLACK,), {}),
         ('circle', ((10, 10), 15), {})]
    )


def test_contains_drawing_operations():
    dl1 = DrawingLayer()
    assert_false(dl1.contains_drawing_operations())

    dl2 = DrawingLayer()
    dl3 = dl1 + dl2
    assert_false(dl3.contains_drawing_operations())

    dl3.select_font('myfont')
    assert_false(dl3.contains_drawing_operations())

    dl3.line((10, 10), (20, 20), color=Color.BLUE)
    assert_true(dl3.contains_drawing_operations())

    dl2.circle((10, 10), 15)
    assert_true(dl3.contains_drawing_operations())
