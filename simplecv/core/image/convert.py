import pygame as pg
try:
    from PIL import Image as PilImage
except:
    import Image as PilImage


def to_pg_surface(img):
    return pg.image.fromstring(img.to_rgb().ndarray.tostring(),
                               img.size, "RGB")


def to_pil_image(img):
    return PilImage.fromstring("RGB", img.size,
                               img.to_rgb().ndarray.tostring())
