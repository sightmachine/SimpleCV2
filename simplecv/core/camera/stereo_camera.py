import cv2
import numpy as np

from simplecv.base import logger
from simplecv.factory import Factory


def set_obj_param(obj, params, param_name):
    param_value = params.get(param_name)
    if param_value is not None:
        setattr(obj, param_name, param_value)


class StereoImage(object):
    """
    **SUMMARY**

    This class is for binaculor Stereopsis. That is exactrating 3D information
    from two differing views of a scene(Image). By comparing the two images,
    the relative depth information can be obtained.

    - Fundamental Matrix : F : a 3 x 3 numpy matrix, is a relationship between
      any two images of the same scene that constrains where the projection
      of points from the scene can occur in both images. see:
      http://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)

    - Homography Matrix : H : a 3 x 3 numpy matrix,

    - ptsLeft : The matched points on the left image.

    - ptsRight : The matched points on the right image.

    -findDisparityMap and findDepthMap - provides 3D information.

    for more information on stereo vision, visit:
        http://en.wikipedia.org/wiki/Computer_stereo_vision

    **EXAMPLE**
    >>> img1 = Image('sampleimages/stereo_view1.png')
    >>> img2 = Image('sampleimages/stereo_view2.png')
    >>> stereoImg = StereoImage(img1, img2)
    >>> stereoImg.find_disparity_map(method="BM", n_disparity=20).show()
    """

    def __init__(self, img_left, img_right):
        super(StereoImage, self).__init__()
        self.image_left = img_left
        self.image_right = img_right
        if self.image_left.size != self.image_right.size:
            logger.warning('Left and Right images should have the same size.')
            return
        else:
            self.size = self.image_left.size
        self.image_3d = None

    def find_fundamental_mat(self, thresh=500.00, min_dist=0.15):
        """
        **SUMMARY**

        This method returns the fundamental matrix F
        such that (P_2).T F P_1 = 0

        **PARAMETERS**

        * *thresh* - The feature quality metric. This can be any value between
                     about 300 and 500. Higher values should return fewer,
                     but higher quality features.
        * *min_dist* - The value below which the feature correspondence is
                       considered a match. This is the distance between two
                       feature vectors. Good values are between 0.05 and 0.3

        **RETURNS**
        Return None if it fails.
        * *F* -  Fundamental matrix as ndarray.
        * *matched_pts1* - the matched points (x, y) in img1
        * *matched_pts2* - the matched points (x, y) in img2

        **EXAMPLE**
        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> F,pts1,pts2 = stereoImg.find_fundamental_mat()

        **NOTE**
        If you deal with the fundamental matrix F directly, be aware of
        (P_2).T F P_1 = 0 where P_2 and P_1 consist of (y, x, 1)
        """

        (kpts1, desc1) = self.image_left._get_raw_keypoints(thresh)
        (kpts2, desc2) = self.image_right._get_raw_keypoints(thresh)

        if desc1 is None or desc2 is None:
            logger.warning("We didn't get any descriptors. Image might "
                           "be too uniform or blurry.")
            return None

        num_pts1 = desc1.shape[0]
        num_pts2 = desc2.shape[0]

        magic_ratio = 1.00
        if num_pts1 > num_pts2:
            magic_ratio = float(num_pts1) / float(num_pts2)

        (idx, dist) = Factory.Image._get_flann_matches(desc1, desc2)
        result = dist.squeeze() * magic_ratio < min_dist

        pts1 = np.array([kpt.pt for kpt in kpts1])
        pts2 = np.array([kpt.pt for kpt in kpts2])

        matched_pts1 = pts1[idx[result]].squeeze()
        matched_pts2 = pts2[result]
        (fnd_mat, mask) = cv2.findFundamentalMat(matched_pts1, matched_pts2,
                                                 method=cv2.FM_LMEDS)

        inlier_ind = mask.nonzero()[0]
        matched_pts1 = matched_pts1[inlier_ind, :]
        matched_pts2 = matched_pts2[inlier_ind, :]

        matched_pts1 = matched_pts1[:, ::-1.00]
        matched_pts2 = matched_pts2[:, ::-1.00]
        return fnd_mat, matched_pts1, matched_pts2

    def find_homography(self, thresh=500.00, min_dist=0.15):
        """
        **SUMMARY**

        This method returns the homography H such that P2 ~ H P1

        **PARAMETERS**

        * *thresh* - The feature quality metric. This can be any value between
                     about 300 and 500. Higher values should return fewer,
                     but higher quality features.
        * *min_dist* - The value below which the feature correspondence is
                       considered a match. This is the distance between two
                       feature vectors. Good values are between 0.05 and 0.3

        **RETURNS**

        Return None if it fails.
        * *H* -  homography as ndarray.
        * *matched_pts1* - the matched points (x, y) in img1
        * *matched_pts2* - the matched points (x, y) in img2

        **EXAMPLE**
        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> H,pts1,pts2 = stereoImg.find_homography()

        **NOTE**
        If you deal with the homography H directly, be aware of P2 ~ H P1
        where P2 and P1 consist of (y, x, 1)
        """

        (kpts1, desc1) = self.image_left._get_raw_keypoints(thresh)
        (kpts2, desc2) = self.image_right._get_raw_keypoints(thresh)

        if desc1 is None or desc2 is None:
            logger.warning("We didn't get any descriptors. Image might be "
                           "too uniform or blurry.")
            return None

        num_pts1 = desc1.shape[0]
        num_pts2 = desc2.shape[0]

        magic_ratio = 1.00
        if num_pts1 > num_pts2:
            magic_ratio = float(num_pts1) / float(num_pts2)

        (idx, dist) = Factory.Image._get_flann_matches(desc1, desc2)
        result = dist.squeeze() * magic_ratio < min_dist

        pts1 = np.array([kpt.pt for kpt in kpts1])
        pts2 = np.array([kpt.pt for kpt in kpts2])

        matched_pts1 = pts1[idx[result]].squeeze()
        matched_pts2 = pts2[result]

        (hmg, mask) = cv2.findHomography(matched_pts1, matched_pts2,
                                         method=cv2.LMEDS)

        inlier_ind = mask.nonzero()[0]
        matched_pts1 = matched_pts1[inlier_ind, :]
        matched_pts2 = matched_pts2[inlier_ind, :]

        matched_pts1 = matched_pts1[:, ::-1.00]
        matched_pts2 = matched_pts2[:, ::-1.00]
        return hmg, matched_pts1, matched_pts2

    def find_disparity_map(self, n_disparity=16, method='BM'):
        """
        The method generates disparity map from set of stereo images.

        **PARAMETERS**

        * *method* :
                 *BM* - Block Matching algorithm, this is a real time
                 algorithm.
                 *SGBM* - Semi Global Block Matching algorithm,
                          this is not a real time algorithm.

        * *n_disparity* - Maximum disparity value. This should be multiple
        of 16
        * *scale* - Scale factor

        **RETURNS**

        Return None if it fails.
        Returns Disparity Map Image

        **EXAMPLE**
        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> disp = stereoImg.find_disparity_map(method="BM")
        """
        gray_left = self.image_left.gray_ndarray
        gray_right = self.image_right.gray_ndarray
        (rows, colums) = self.size
        #scale = int(self.image_left.depth)
        if n_disparity % 16 != 0:
            if n_disparity < 16:
                n_disparity = 16
            n_disparity = (n_disparity / 16) * 16
        try:
            if method == 'BM':
                sbm = cv2.StereoBM(cv2.cv.CV_STEREO_BM_BASIC,
                                   ndisparities=n_disparity,
                                   SADWindowSize=41)

                dsp = sbm.compute(gray_left, gray_right)
            elif method == 'SGBM':
                ssgbm = cv2.StereoSGBM(minDisparity=0,
                                       numDisparities=n_disparity,
                                       SADWindowSize=41,
                                       preFilterCap=31,
                                       disp12MaxDiff=1,
                                       fullDP=False,
                                       P1=8 * 1 * 41 * 41,
                                       P2=32 * 1 * 41 * 41,
                                       uniquenessRatio=15)
                dsp = ssgbm.compute(gray_left, gray_right)
            else:
                logger.warning("Unknown method. Choose one method amoung "
                               "BM or SGBM or GC !")
                return None

            dsp_visual = cv2.normalize(dsp, alpha=0, beta=256,
                                       norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_8U)
            return Factory.Image(dsp_visual)
        except Exception:
            logger.warning("Error in computing the Disparity Map, may be "
                           "due to the Images are stereo in nature.")
            return None

    def eline(self, point, fnd_mat, which_image):
        """
        **SUMMARY**

        This method returns, line feature object.

        **PARAMETERS**

        * *point* - Input point (x, y)
        * *fnd_mat* - Fundamental matrix.
        * *which_image* - Index of the image (1 or 2) that contains the point

        **RETURNS**

        epipolar line, in the form of line feature object.

        **EXAMPLE**

        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> F,pts1,pts2 = stereoImg.find_fundamental_mat()
        >>> point = pts2[0]
        >>> #find corresponding Epipolar line in the left image.
        >>> epiline = mapper.eline(point, F, 1)
        """
        from simplecv.features.detection import Line

        # TODO: cv2.computeCorrespondEpilines will be available in OpenCV 3.0
        # pts1 = (0, 0)
        # pts2 = self.size
        # pt_mat = np.zeros((1, 1), dtype=np.float32)
        # # OpenCV seems to use (y, x) coordinate.
        # pt_mat[0, 0] = (point[1], point[0])
        # line = np.zeros((1, 1), dtype=np.float32)
        # cv2.computeCorrespondEpilines(pt_mat, which_image, fnd_mat, line)
        # line_np_array = np.array(line).squeeze()
        # line_np_array = line_np_array[[1.00, 0, 2]]
        # pts1 = (pts1[0], (-line_np_array[2] - line_np_array[0] * pts1[0])
        #         / line_np_array[1])
        # pts2 = (pts2[0], (-line_np_array[2] - line_np_array[0] * pts2[0])
        #         / line_np_array[1])
        # if which_image == 1:
        #     return Line(self.image_left, [pts1, pts2])
        # elif which_image == 2:
        #     return Line(self.image_right, [pts1, pts2])
        ######################################################################

        # According to http://ai.stanford.edu/~mitul/cs223b/draw_epipolar.m

        def epipole_svd(f):
            v = cv2.SVDecomp(f)[2]
            return v[-1] / v[-1, -1]

        if which_image == 1:
            e1c2 = epipole_svd(fnd_mat)
            m = float(point[1] - e1c2[0]) / float(point[0] - e1c2[1])
            c = e1c2[0] - m * e1c2[1]
            pts1 = (0, c)
            pts2 = (self.image_left.width, m * self.image_left.width + c)
            return Line(self.image_left, [pts1, pts2])

        elif which_image == 2:
            e2c1 = epipole_svd(fnd_mat.T)
            m = float(point[1] - e2c1[0]) / float(point[0] - e2c1[1])
            c = e2c1[0] - m * e2c1[1]
            pts1 = (0, c)
            pts2 = (self.image_right.width, m * self.image_right.width + c)
            return Line(self.image_right, [pts1, pts2])
        else:
            logger.warn("Incorrect Image number passed. Returning None.")
            return None

    def project_point(self, point, hmg, which_image):
        """
        **SUMMARY**

        This method returns the corresponding point (x, y)

        **PARAMETERS**

        * *point* - Input point (x, y)
        * *which_image* - Index of the image (1 or 2) that contains the point
        * *hmg* - Homography that can be estimated
                  using StereoCamera.find_homography()

        **RETURNS**

        Corresponding point (x, y) as tuple

        **EXAMPLE**

        >>> img1 = Image("sampleimages/stereo_view1.png")
        >>> img2 = Image("sampleimages/stereo_view2.png")
        >>> stereoImg = StereoImage(img1,img2)
        >>> F,pts1,pts2 = stereoImg.find_fundamental_mat()
        >>> point = pts2[0]
        >>> #finds corresponding  point in the left image.
        >>> projectPoint = stereoImg.project_point(point, H, 1)
        """

        hmg = np.matrix(hmg)
        point = np.matrix((point[1], point[0], 1.00))
        if which_image == 1.00:
            corres_pt = hmg * point.T
        else:
            corres_pt = np.linalg.inv(hmg) * point.T
        corres_pt = corres_pt / corres_pt[2]
        return float(corres_pt[1]), float(corres_pt[0])

    def get_3d_image(self, rpj_mat, method="BM", state=None):
        """
        **SUMMARY**

        This method returns the 3D depth image using reprojectImageTo3D method.

        **PARAMETERS**

        * *rpj_mat* - reprojection Matrix (disparity to depth matrix)
        * *method* - Stereo Correspondonce method to be used.
                   - "BM" - Stereo BM
                   - "SGBM" - Stereo SGBM
        * *state* - dictionary corresponding to parameters of
                    stereo correspondonce.
                    SADWindowSize - odd int
                    numberOfDisparities - int
                    minDisparity  - int
                    preFilterCap - int
                    preFilterType - int (only BM)
                    speckleRange - int
                    speckleWindowSize - int
                    P1 - int (only SGBM)
                    P2 - int (only SGBM)
                    fullDP - Bool (only SGBM)
                    uniquenessRatio - int
                    textureThreshold - int (only BM)

        **RETURNS**

        SimpleCV.Image representing 3D depth Image
        also StereoImage.Image3D gives OpenCV 3D Depth Image of CV_32F type.

        **EXAMPLE**

        >>> lImage = Image("l.jpg")
        >>> rImage = Image("r.jpg")
        >>> stereo = StereoImage(lImage, rImage)
        >>> rpj_mat = cv.Load("Q.yml")
        >>> stereo.get_3d_image(rpj_mat).show()

        >>> state = {"SADWindowSize":9, "numberOfDisparities":112}
        >>> stereo.get_3d_image(rpj_mat, "BM", state).show()
        >>> stereo.get_3d_image(rpj_mat, "SGBM", state).show()
        """
        gray_left = self.image_left.gray_ndarray
        gray_right = self.image_right.gray_ndarray

        if method == "BM":
            sbm = cv2.StereoBM()
            if not state:
                state = {"SADWindowSize": 9, "numberOfDisparities": 112,
                         "preFilterType": 1, "speckleWindowSize": 0,
                         "minDisparity": -39, "textureThreshold": 507,
                         "preFilterCap": 61, "uniquenessRatio": 0,
                         "speckleRange": 8, "preFilterSize": 5}

            set_obj_param(sbm, state, "SADWindowSize")
            set_obj_param(sbm, state, "preFilterCap")
            set_obj_param(sbm, state, "minDisparity")
            set_obj_param(sbm, state, "numberOfDisparities")
            set_obj_param(sbm, state, "uniquenessRatio")
            set_obj_param(sbm, state, "speckleRange")
            set_obj_param(sbm, state, "speckleWindowSize")
            set_obj_param(sbm, state, "textureThreshold")
            set_obj_param(sbm, state, "preFilterType")

            disparity = sbm.compute(gray_left, gray_right, cv2.CV_32F)

        elif method == "SGBM":
            ssgbm = cv2.StereoSGBM()
            if not state:
                state = {"SADWindowSize": 9, "numberOfDisparities": 96,
                         "minDisparity": -21, "speckleWindowSize": 0,
                         "preFilterCap": 61, "uniquenessRatio": 7,
                         "speckleRange": 8, "disp12MaxDiff": 1,
                         "fullDP": False}
                set_obj_param(ssgbm, state, "disp12MaxDiff")

            set_obj_param(ssgbm, state, "SADWindowSize")
            set_obj_param(ssgbm, state, "preFilterCap")
            set_obj_param(ssgbm, state, "minDisparity")
            set_obj_param(ssgbm, state, "numberOfDisparities")
            set_obj_param(ssgbm, state, "P1")
            set_obj_param(ssgbm, state, "P2")
            set_obj_param(ssgbm, state, "uniquenessRatio")
            set_obj_param(ssgbm, state, "speckleRange")
            set_obj_param(ssgbm, state, "speckleWindowSize")
            set_obj_param(ssgbm, state, "fullDP")

            disparity = ssgbm.compute(gray_left, gray_right)
        else:
            logger.warn("Unknown method. Returning None")
            return None

        if not isinstance(rpj_mat, np.ndarray):
            rpj_mat = np.array(rpj_mat)
        if not isinstance(disparity, np.ndarray):
            disparity = np.array(disparity)
        image_3d = cv2.reprojectImageTo3D(disparity, rpj_mat,
                                          ddepth=cv2.CV_32F)
        image_3d_normalize = cv2.normalize(image_3d, alpha=0, beta=255,
                                           norm_type=cv2.cv.CV_MINMAX,
                                           dtype=cv2.CV_8UC3)
        ret_value = Factory.Image(image_3d_normalize)

        self.image_3d = image_3d
        return ret_value

    def get_3d_image_from_disparity(self, disparity, rpj_mat):
        """
        **SUMMARY**

        This method returns the 3D depth image using reprojectImageTo3D method.

        **PARAMETERS**
        * *disparity* - Disparity Image
        * *rpj_mat* - reprojection Matrix (disparity to depth matrix)

        **RETURNS**

        SimpleCV.Image representing 3D depth Image
        also StereoCamera.Image3D gives OpenCV 3D Depth Image of CV_32F type.

        **EXAMPLE**

        >>> lImage = Image("l.jpg")
        >>> rImage = Image("r.jpg")
        >>> stereo = StereoCamera()
        >>> rpj_mat = cv.Load("Q.yml")
        >>> disp = stereo.find_disparity_map()
        >>> stereo.get_3d_image_from_disparity(disp, rpj_mat)
        """

        if not isinstance(rpj_mat, np.ndarray):
            rpj_mat = np.array(rpj_mat)
        disparity = disparity.get_numpy_cv2()
        image_3d = cv2.reprojectImageTo3D(disparity, rpj_mat,
                                          ddepth=cv2.CV_32F)
        image_3d_normalize = cv2.normalize(image_3d, alpha=0, beta=255,
                                           norm_type=cv2.cv.CV_MINMAX,
                                           dtype=cv2.CV_8UC3)
        ret_value = Factory.Image(image_3d_normalize)

        self.image_3d = image_3d
        return ret_value


class StereoCamera(object):
    """
    Stereo Camera is a class dedicated for calibration stereo camera.
    It also has functionalites for rectification and getting undistorted
    Images.

    This class can be used to calculate various parameters
    related to both the camera's :
      -> Camera Matrix
      -> Distortion coefficients
      -> Rotation and Translation matrix
      -> Rectification transform (rotation matrix)
      -> Projection matrix in the new (rectified) coordinate systems
      -> Disparity-to-depth mapping matrix (Q)
    """

    def __init__(self):
        pass

    def stereo_calibration(self, cam_left, cam_right, nboards=30,
                           chessboard=(8, 5), grid_size=0.027,
                           win_size=(352, 288)):
        """

        **SUMMARY**

        Stereo Calibration is a way in which you obtain the parameters that
        will allow you to calculate 3D information of the scene.
        Once both the camera's are initialized.
        Press [Space] once chessboard is identified in both the camera's.
        Press [esc] key to exit the calibration process.

        **PARAMETERS**

        * cam_left - Left camera index.
        * cam_right - Right camera index.
        * nboards - Number of samples or multiple views of the chessboard in
                    different positions and orientations with your stereo
                    camera
        * chessboard - A tuple of Cols, Rows in
                       the chessboard (used for calibration).
        * grid_size - chessboard grid size in real units
        * win_size - This is the window resolution.

        **RETURNS**

        A tuple of the form (cm1, cm2, d1, d2, r, t, e, f) on success
        cm1 - Camera Matrix for left camera,
        cm2 - Camera Matrix for right camera,
        d1 - Vector of distortion coefficients for left camera,
        d2 - Vector of distortion coefficients for right camera,
        r - Rotation matrix between the left and the right
            camera coordinate systems,
        t - Translation vector between the left and the right
            coordinate systems of the cameras,
        e - Essential matrix,
        f - Fundamental matrix

        **EXAMPLE**

        >>> StereoCam = StereoCamera()
        >>> calibration = StereoCam.stereo_calibration(1,2,nboards=40)

        **Note**

        Press space to capture the images.

        """
        count = 0
        left = "Left"
        right = "Right"

        capture_left = cv2.VideoCapture()
        ret = capture_left.open(cam_left)
        capture_left.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, win_size[0])
        capture_left.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, win_size[1])
        _, frame_left = capture_left.read()
        cv2.findChessboardCorners(frame_left, chessboard)

        capture_right = cv2.VideoCapture()
        ret = capture_right.open(cam_right)
        capture_right.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, win_size[0])
        capture_right.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, win_size[1])
        _, frame_right = capture_right.read()
        cv2.findChessboardCorners(frame_right, chessboard)

        cols = nboards * chessboard[0] * chessboard[1]
        image_points1 = np.zeros((cols, 2), dtype=np.float64)
        image_points2 = np.zeros((cols, 2), dtype=np.float64)
        object_points = np.zeros((cols, 3), dtype=np.float64)

        while True:
            _, frame_left = capture_left.read()
            frame_left = cv2.flip(frame_left, 1)
            _, frame_right = capture_right.read()
            frame_right = cv2.flip(frame_right, 1)
            k = cv2.waitKey(3)

            res1, cor1 = cv2.findChessboardCorners(frame_left, chessboard)
            if res1:
                cv2.drawChessboardCorners(frame_left, chessboard, cor1, res1)
                cv2.imshow(left, frame_left)

            res2, cor2 = cv2.findChessboardCorners(frame_right, chessboard)
            if cor2:
                cv2.drawChessboardCorners(frame_right, chessboard, cor2, res2)
                cv2.imshow(right, frame_right)

            cbrd_mlt = chessboard[0] * chessboard[1]
            if res1 and res2 and k == 0x20:
                print count
                for i in range(0, len(cor1[1])):
                    image_points1[count * cbrd_mlt + i] = (cor1[i][0],
                                                           cor1[i][1])
                    image_points2[count * cbrd_mlt + i] = (cor2[i][0],
                                                           cor2[i][1])

                count += 1

                if count == nboards:
                    cv2.destroyAllWindows()
                    for i in range(nboards):
                        for j in range(chessboard[1]):
                            for k in range(chessboard[0]):
                                object_points[i * cbrd_mlt + j
                                              * chessboard[0] + k] = \
                                    (k * grid_size, j * grid_size, 0)

                    print "Running stereo calibration..."
                    rtval, cm1, d1, cm2, d2, r, t, e, f = cv2.stereoCalibrate(
                        object_points, image_points1, image_points2,
                        win_size, flags=cv2.CALIB_SAME_FOCAL_LENGTH |
                        cv2.CALIB_ZERO_TANGENT_DIST)
                    if rtval:
                        print "Done."
                        return cm1, cm2, d1, d2, r, t, e, f
                    else:
                        print "Failed."
                        return None

            cv2.imshow(left, frame_left)
            cv2.imshow(right, frame_right)
            if k == 0x1b:
                print "ESC pressed. Exiting. "\
                      "WARNING: NOT ENOUGH CHESSBOARDS FOUND YET"
                cv2.destroyAllWindows()
                break

    def save_calibration(self, calibration=None, fname="Stereo", cdir="."):
        """

        **SUMMARY**

        save_calibration is a method to save the StereoCalibration parameters
        such as CM1, CM2, D1, D2, R, T, E, F of stereo pair.
        This method returns True on success and saves the calibration
        in the following format.
        StereoCM1.txt
        StereoCM2.txt
        StereoD1.txt
        StereoD2.txt
        StereoR.txt
        StereoT.txt
        StereoE.txt
        StereoF.txt

        **PARAMETERS**

        calibration - is a tuple os the form (CM1, CM2, D1, D2, R, T, E, F)
        CM1 -> Camera Matrix for left camera,
        CM2 -> Camera Matrix for right camera,
        D1 -> Vector of distortion coefficients for left camera,
        D2 -> Vector of distortion coefficients for right camera,
        R -> Rotation matrix between the left and the right
             camera coordinate systems,
        T -> Translation vector between the left and the right
             coordinate systems of the cameras,
        E -> Essential matrix,
        F -> Fundamental matrix


        **RETURNS**

        return True on success and saves the calibration files.

        **EXAMPLE**

        >>> StereoCam = StereoCamera()
        >>> calibration = StereoCam.stereo_calibration(1, 2, nboards=40)
        >>> StereoCam.save_calibration(calibration, fname="Stereo1")
        """
        (cm1, cm2, d1, d2, r, t, e, f) = calibration
        np.save("{0}/{1}CM1".format(cdir, fname), cm1)
        np.save("{0}/{1}CM2".format(cdir, fname), cm2)
        np.save("{0}/{1}D1".format(cdir, fname), d1)
        np.save("{0}/{1}D2".format(cdir, fname), d2)
        np.save("{0}/{1}R".format(cdir, fname), r)
        np.save("{0}/{1}T".format(cdir, fname), t)
        np.save("{0}/{1}E".format(cdir, fname), e)
        np.save("{0}/{1}F".format(cdir, fname), f)
        logger.debug("Calibration parameters written to directory"
                     " '{0}'.".format(cdir))
        return True

    def load_calibration(self, fname="Stereo", dir="."):
        """

        **SUMMARY**

        load_calibration is a method to load the StereoCalibration parameters
        such as CM1, CM2, D1, D2, R, T, E, F of stereo pair.
        This method loads from calibration files and return calibration
        on success else return false.

        **PARAMETERS**

        fname - is the prefix of the calibration files.
        dir - is the directory in which files are present.

        **RETURNS**

        a tuple of the form (CM1, CM2, D1, D2, R, T, E, F) on success.
        CM1 - Camera Matrix for left camera
        CM2 - Camera Matrix for right camera
        D1 - Vector of distortion coefficients for left camera
        D2 - Vector of distortion coefficients for right camera
        R - Rotation matrix between the left and the right
            camera coordinate systems
        T - Translation vector between the left and the right
            coordinate systems of the cameras
        E - Essential matrix
        F - Fundamental matrix
        else returns false

        **EXAMPLE**

        >>> StereoCam = StereoCamera()
        >>> loadedCalibration = StereoCam.load_calibration(fname="Stereo1")

        """
        cm1 = np.load("{0}/{1}CM1.npy".format(dir, fname))
        cm2 = np.load("{0}/{1}CM2.npy".format(dir, fname))
        d1 = np.load("{0}/{1}D1.npy".format(dir, fname))
        d2 = np.load("{0}/{1}D2.npy".format(dir, fname))
        r = np.load("{0}/{1}R.npy".format(dir, fname))
        t = np.load("{0}/{1}T.npy".format(dir, fname))
        e = np.load("{0}/{1}E.npy".format(dir, fname))
        f = np.load("{0}/{1}F.npy".format(dir, fname))
        logger.debug("Calibration files loaded from dir '{0}'.".format(dir))
        return cm1, cm2, d1, d2, r, t, e, f

    def stereo_rectify(self, calibration=None, win_size=(352, 288)):
        """

        **SUMMARY**

        Computes rectification transforms for each head
        of a calibrated stereo camera.

        **PARAMETERS**

        calibration - is a tuple os the form (CM1, CM2, D1, D2, R, T, E, F)
        CM1 - Camera Matrix for left camera,
        CM2 - Camera Matrix for right camera,
        D1 - Vector of distortion coefficients for left camera,
        D2 - Vector of distortion coefficients for right camera,
        R - Rotation matrix between the left and the right
            camera coordinate systems,
        T - Translation vector between the left and the right
            coordinate systems of the cameras,
        E - Essential matrix,
        F - Fundamental matrix

        **RETURNS**

        On success returns a a tuple of the format -> (R1, R2, P1, P2, Q, roi)
        R1 - Rectification transform (rotation matrix) for the left camera.
        R2 - Rectification transform (rotation matrix) for the right camera.
        P1 - Projection matrix in the new (rectified) coordinate systems
             for the left camera.
        P2 - Projection matrix in the new (rectified) coordinate systems
             for the right camera.
        Q - disparity-to-depth mapping matrix.

        **EXAMPLE**

        >>> StereoCam = StereoCamera()
        >>> calibration = StereoCam.load_calibration(fname="Stereo1")
        >>> rectification = StereoCam.stereo_rectify(calibration)

        """
        (cm1, cm2, d1, d2, r, t, e, f) = calibration

        logger.debug("Running stereo rectification...")

        result = cv2.stereoRectify(cm1, d1, cm2, d2, win_size, r, t)
        r1, r2, p1, p2, q, left_roi, right_roi = result

        roi = []
        roi.append(max(left_roi[0], right_roi[0]))
        roi.append(max(left_roi[1], right_roi[1]))
        roi.append(min(left_roi[2], right_roi[2]))
        roi.append(min(left_roi[3], right_roi[3]))
        logger.debug("Done stereo rectification.")
        return r1, r2, p1, p2, q, roi

    def get_images_undistort(self, img_left, img_right, calibration,
                             rectification, win_size=(352, 288)):
        """
        **SUMMARY**
        Rectify two images from the calibration and rectification parameters.

        **PARAMETERS**
        * *img_left* - Image captured from left camera
                       and needs to be rectified.
        * *img_right* - Image captures from right camera
                       and need to be rectified.
        * *calibration* - A calibration tuple of the format
                          (CM1, CM2, D1, D2, R, T, E, F)
        * *rectification* - A rectification tuple of the format
                           (R1, R2, P1, P2, Q, roi)

        **RETURNS**
        returns rectified images in a tuple -> (img_left,img_right)

        **EXAMPLE**
        >>> StereoCam = StereoCamera()
        >>> calibration = StereoCam.load_calibration(fname="Stereo1")
        >>> rectification = StereoCam.stereo_rectify(calibration)
        >>> img_left = cam_left.get_image()
        >>> img_right = cam_right.get_image()
        >>> rectLeft,rectRight = StereoCam.get_images_undistort(img_left,
                                       img_right,calibration,rectification)
        """
        img_left = img_left.get_matrix()
        img_right = img_right.get_matrix()
        (cm1, cm2, d1, d2, r, t, e, f) = calibration
        (r1, r2, p1, p2, q, roi) = rectification

        map1x, map1y = cv2.initUndistortRectifyMap(cm1, d1, r1, p1, win_size,
                                                   cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(cm2, d2, r2, p2, win_size,
                                                   cv2.CV_32FC1)

        dst1 = cv2.remap(img_left, map1x, map1y, cv2.INTER_LINEAR)
        dst2 = cv2.remap(img_right, map2x, map2y, cv2.INTER_LINEAR)
        return Factory.Image(dst1), Factory.Image(dst2)

    def get_3d_image(self, left_index, right_index, rpj_mat, method="BM",
                     state=None):
        """
        **SUMMARY**

        This method returns the 3D depth image using reprojectImageTo3D method.

        **PARAMETERS**

        * *left_index* - Index of left camera
        * *right_index* - Index of right camera
        * *rpj_mat* - reprojection Matrix (disparity to depth matrix)
        * *method* - Stereo Correspondonce method to be used.
                   - "BM" - Stereo BM
                   - "SGBM" - Stereo SGBM
        * *state* - dictionary corresponding to parameters of
                    stereo correspondonce.
                    SADWindowSize - odd int
                    n_disparity - int
                    min_disparity  - int
                    preFilterCap - int
                    preFilterType - int (only BM)
                    speckleRange - int
                    speckleWindowSize - int
                    P1 - int (only SGBM)
                    P2 - int (only SGBM)
                    fullDP - Bool (only SGBM)
                    uniquenessRatio - int
                    textureThreshold - int (only BM)


        **RETURNS**

        SimpleCV.Image representing 3D depth Image
        also StereoCamera.Image3D gives OpenCV 3D Depth Image of CV_32F type.

        **EXAMPLE**

        >>> lImage = Image("l.jpg")
        >>> rImage = Image("r.jpg")
        >>> stereo = StereoCamera()
        >>> Q = cv.Load("Q.yml")
        >>> stereo.get_3d_image(1, 2, Q).show()

        >>> state = {"SADWindowSize":9, "n_disparity":112, "min_disparity":-39}
        >>> stereo.get_3d_image(1, 2, Q, "BM", state).show()
        >>> stereo.get_3d_image(1, 2, Q, "SGBM", state).show()
        """

        cam_left = cv2.VideoCapture(left_index)
        cam_right = cv2.VideoCapture(right_index)
        if cam_left.isOpened():
            _, img_left = cam_left.read()
        else:
            logger.warn("Unable to open left camera")
            return None
        if cam_right.isOpened():
            _, img_right = cam_right.read()
        else:
            logger.warn("Unable to open right camera")
            return None
        img_left = Factory.Image(img_left)
        img_right = Factory.Image(img_right)

        del cam_left
        del cam_right

        stereo_images = StereoImage(img_left, img_right)
        image_3d_normalize = stereo_images.get_3d_image(rpj_mat, method, state)
        #self.image_3d = stereo_images.image_3d
        return image_3d_normalize
