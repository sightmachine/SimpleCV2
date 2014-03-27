from simplecv.camera import Camera, VirtualCamera


testoutput = "../data/sampleimages/cam.jpg"


def test_virtual_camera_constructor():
    mycam = VirtualCamera(testoutput, 'image')

    props = mycam.get_all_properties()

    for i in props.keys():
        print str(i) + ": " + str(props[i]) + "\n"


def test_camera_image():
    mycam = Camera(0)

    img = mycam.get_image()
    img.save(testoutput)


def test_camera_multiple_instances():
    cam1 = Camera()
    img1 = cam1.get_image()
    cam2 = Camera()
    img2 = cam2.get_image()

    if not cam1 or not cam2 or not img1 or not img2:
        assert False

    cam3 = Camera(0)  # assuming the default camera index is 0
    img3 = cam3.get_image()

    if not cam3 or not img3:
        assert False
