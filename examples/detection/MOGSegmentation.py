from simplecv import Camera, Display
from simplecv.color import Color
from simplecv.segmentation.mog_segmentation import MOGSegmentation

mog = MOGSegmentation(history=200, mixtures=5, bg_ratio=0.3, noise_sigma=16,
                      learningrate=0.3)

cam = Camera()

disp = Display()

while disp.is_not_done():
    frame = cam.get_image()

    mog.add_image(frame)

    segmentedImage = mog.get_segmented_image()
    blobs = mog.get_segmented_blobs()
    for blob in blobs:
        segmentedImage.dl().circle((blob.x, blob.y), 10, Color.RED)

    segmentedImage.save(disp)
