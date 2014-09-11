import cv2

from simplecv.base import logger
from simplecv.core.image import image_method
from simplecv.tracking.cam_shift_tracker import camshiftTracker
from simplecv.tracking.lk_tracker import lkTracker
from simplecv.tracking.mf_tracker import mfTracker
from simplecv.tracking.surf_tracker import surfTracker
from simplecv.tracking.track_set import TrackSet


@image_method
def track(self, method="CAMShift", ts=None, img=None, bb=None, **kwargs):
    """
    **DESCRIPTION**

    Tracking the object surrounded by the bounding box in the given
    image or TrackSet.

    **PARAMETERS**

    * *method* - str - The Tracking Algorithm to be applied
    * *ts* - TrackSet - SimpleCV.Features.TrackSet.
    * *img* - Image - Image to be tracked or list - List of Images to be
      tracked.
    * *bb* - tuple - Bounding Box tuple (x, y, w, h)


    **Optional Parameters**

    *CAMShift*

    CAMShift Tracker is based on mean shift thresholding algorithm which is
    combined with an adaptive region-sizing step. Histogram is calcualted
    based on the mask provided. If mask is not provided, hsv transformed
    image of the provided image is thresholded using inRange function
    (band thresholding).

    lower HSV and upper HSV values are used inRange function. If the user
    doesn't provide any range values, default range values are used.

    Histogram is back projected using previous images to get an appropriate
    image and it passed to camshift function to find the object in the
    image. Users can decide the number of images to be used in back
    projection by providing num_frames.

    lower - Lower HSV value for inRange thresholding. tuple of (H, S, V).
            Default : (0, 60, 32)
    upper - Upper HSV value for inRange thresholding. tuple of (H, S, V).
            Default: (180, 255, 255)
    mask - Mask to calculate Histogram. It's better if you don't provide
           one. Default: calculated using above thresholding ranges.
    num_frames - number of frames to be backtracked. Default: 40

    *LK*

    LK Tracker is based on Optical Flow method. In brief, optical flow can
    be defined as the apparent motion of objects caused by the relative
    motion between an observer and the scene. (Wikipedia).

    LK Tracker first finds some good feature points in the given bounding
    box in the image. These are the tracker points. In consecutive frames,
    optical flow of these feature points is calculated. Users can limit the
    number of feature points by provideing maxCorners and qualityLevel.
    Number of features will always be less than maxCorners. These feature
    points are calculated using Harris Corner detector. It returns a matrix
    with each pixel having some quality value. Only good features are used
    based upon the qualityLevel provided. better features have better
    quality measure and hence are more suitable to track.

    Users can set minimum distance between each features by providing
    minDistance.

    LK tracker finds optical flow using a number of pyramids and users
    can set this number by providing maxLevel and users can set size of the
    search window for Optical Flow by setting winSize.

    docs from http://docs.opencv.org/
    maxCorners - Maximum number of corners to return in
                 goodFeaturesToTrack. If there are more corners than are
                 found, the strongest of them is returned. Default: 4000
    qualityLevel - Parameter characterizing the minimal accepted quality of
                   image corners. The parameter value is multiplied by the
                   best corner quality measure, which is the minimal
                   eigenvalue or the Harris function response. The corners
                   with the quality measure less than the product are
                   rejected. For example, if the best corner has the
                   quality measure = 1500,  and the qualityLevel=0.01 ,
                   then all the corners with the quality measure less than
                   15 are rejected. Default: 0.08
    minDistance - Minimum possible Euclidean distance between the returned
                  corners. Default: 2
    blockSize - Size of an average block for computing a derivative
                covariation matrix over each pixel neighborhood. Default: 3
    winSize - size of the search window at each pyramid level.
              Default: (10, 10)
    maxLevel - 0-based maximal pyramid level number; if set to 0, pyramids
               are not used (single level), Default: 10 if set to 1, two
               levels are used, and so on

    *SURF*

    SURF based tracker finds keypoints in the template and computes the
    descriptor. The template is chosen based on the bounding box provided
    with the first image. The image is cropped and stored as template. SURF
    keypoints are found and descriptor is computed for the template and
    stored.

    SURF keypoints are found in the image and its descriptor is computed.
    Image keypoints and template keypoints are matched using K-nearest
    neighbor algorithm. Matched keypoints are filtered according to the knn
    distance of the points. Users can set this criteria by setting
    distance. Density Based Clustering algorithm (DBSCAN) is applied on
    the matched keypoints to filter out points that are in background.
    DBSCAN creates a cluster of object points anc background points. These
    background points are discarded. Users can set certain parameters for
    DBSCAN which are listed below.

    K-means is applied on matched KeyPoints with k=1 to find the center of
    the cluster and then bounding box is predicted based upon the position
    of all the object KeyPoints.

    eps_val - eps for DBSCAN. The maximum distance between two samples for
      them to be considered as in the same neighborhood. default: 0.69
    min_samples - min number of samples in DBSCAN. The number of samples
      in a neighborhood for a point to be considered as a core point.
      default: 5
    distance - thresholding KNN distance of each feature. if
    KNN distance > distance, point is discarded. default: 100

    *MFTrack*

    Median Flow tracker is similar to LK tracker (based on Optical Flow),
    but it's more advanced, better and faster.

    In MFTrack, tracking points are decided based upon the number of
    horizontal and vertical points and window size provided by the user.
    Unlike LK Tracker, good features are not found which saves a huge
    amount of time.

    feature points are selected symmetrically in the bounding box.
    Total number of feature points to be tracked = numM * numN.

    If the width and height of bounding box is 200 and 100 respectively,
    and numM = 10 and numN = 10, there will be 10 points in the bounding
    box equally placed(10 points in 200 pixels) in each row. and 10 equally
    placed points (10 points in 100 pixels) in each column. So total number
    of tracking points = 100.

    numM > 0
    numN > 0 (both may not be equal)

    users can provide a margin around the bounding box that will be
    considered to place feature points and calculate optical flow.
    Optical flow is calculated from frame1 to frame2 and from frame2 to
    frame1. There might be some points which give inaccurate optical flow,
    to eliminate these points the above method is used. It is called
    forward-backward error tracking. Optical Flow seach window size can be
    set usung winsize_lk.

    For each point, comparision is done based on the quadratic area around
    it. The length of the square window can be set using winsize.

    numM        - Number of points to be tracked in the bounding box
                  in height direction.
                  default: 10

    numN        - Number of points to be tracked in the bounding box
                  in width direction.
                  default: 10

    margin      - Margin around the bounding box.
                  default: 5

    winsize_lk  - Optical Flow search window size.
                  default: 4

    winsize     - Size of quadratic area around the point which is
                  compared. default: 10


    Available Tracking Methods

     - CamShift
     - LK
     - SURF
     - MFTrack


    **RETURNS**

    SimpleCV.Features.TrackSet

    Returns a TrackSet with all the necessary attributes.

    **HOW TO**

    >>> ts = img.track("camshift", img=img1, bb=bb)


    Here TrackSet is returned. All the necessary attributes will be
    included in the trackset. After getting the trackset you need not
    provide the bounding box or image. You provide TrackSet as parameter
    to track(). Bounding box and image will be taken from the trackset.
    So. now

    >>> ts = new_img.track("camshift", ts)

    The new Tracking feature will be appended to the given trackset and
    that will be returned.
    So, to use it in loop::

      img = cam.getImage()
      bb = (img.width/4,img.height/4,img.width/4,img.height/4)
      ts = img.track(img=img, bb=bb)
      while (True):
          img = cam.getImage()
          ts = img.track("camshift", ts=ts)

      ts = []
      while (some_condition_here):
          img = cam.getImage()
          ts = img.track("camshift",ts,img0,bb)


    now here in first loop iteration since ts is empty, img0 and bb will
    be considered. New tracking object will be created and added in ts
    (TrackSet) After first iteration, ts is not empty and hence the
    previous image frames and bounding box will be taken from ts and img0
    and bb will be ignored.

    # Instead of loop, give a list of images to be tracked.

    ts = []
    imgs = [img1, img2, img3, ..., imgN]
    ts = img0.track("camshift", ts, imgs, bb)
    ts.drawPath()
    ts[-1].image.show()

    Using Optional Parameters:

    for CAMShift

    >>> ts = []
    >>> ts = img.track("camshift", ts, img1, bb, lower=(40, 100, 100),
        ...            upper=(100, 250, 250))

    You can provide some/all/None of the optional parameters listed
    for CAMShift.

    for LK

    >>> ts = []
    >>> ts = img.track("lk", ts, img1, bb, maxCorners=4000,
        ...            qualityLevel=0.5, minDistance=3)

    You can provide some/all/None of the optional parameters listed for LK.

    for SURF

    >>> ts = []
    >>> ts = img.track("surf", ts, img1, bb, eps_val=0.7, min_samples=8,
        ...            distance=200)

    You can provide some/all/None of the optional parameters listed
    for SURF.

    for MFTrack
    >>> ts = []
    >>> ts = img.track("mftrack", ts, img1, bb, numM=12, numN=12,
        ...            winsize=15)

    You can provide some/all/None of the optional parameters listed for
    MFTrack.

    Check out Tracking examples provided in the SimpleCV source code.

    READ MORE:

    CAMShift Tracker:
    Uses meanshift based CAMShift thresholding technique. Blobs and objects
    with single tone or tracked very efficiently. CAMshift should be
    preferred if you are trying to track faces. It is optimized to track
    faces.

    LK (Lucas Kanade) Tracker:
    It is based on LK Optical Flow. It calculates Optical flow in frame1
    to frame2 and also in frame2 to frame1 and using back track error,
    filters out false positives.

    SURF based Tracker:
    Matches keypoints from the template image and the current frame.
    flann based matcher is used to match the keypoints.
    Density based clustering is used classify points as in-region
    (of bounding box) and out-region points. Using in-region points, new
    bounding box is predicted using k-means.

    Median Flow Tracker:

    Media Flow Tracker is the base tracker that is used in OpenTLD. It is
    based on Optical Flow. It calculates optical flow of the points in the
    bounding box from frame 1 to frame 2 and from frame 2 to frame 1 and
    using back track error, removes false positives. As the name suggests,
    it takes the median of the flow, and eliminates points.
    """
    if ts is None and img is None:
        print "Invalid Input. Must provide FeatureSet or Image"
        return None

    if ts is None and bb is None:
        print "Invalid Input. Must provide Bounding Box with Image"
        return None

    if not ts:
        ts = TrackSet()
    else:
        img = ts[-1].image
        bb = ts[-1].bb

    if type(img) == list:
        ts = self.track(method, ts, img[0], bb, **kwargs)
        for i in img:
            ts = i.track(method, ts, **kwargs)
        return ts

    # Issue #256 - (Bug) Memory management issue due to too many number
    # of images.
    nframes = 300
    if 'nframes' in kwargs:
        nframes = kwargs['nframes']

    if len(ts) > nframes:
        ts.trimList(50)

    if method.lower() == "camshift":
        track = camshiftTracker(self, bb, ts, **kwargs)
        ts.append(track)

    elif method.lower() == "lk":
        track = lkTracker(self, bb, ts, img, **kwargs)
        ts.append(track)

    elif method.lower() == "surf":
        try:
            from scipy.spatial import distance as dist
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.warning("sklearn required")
            return None
        if not hasattr(cv2, "FeatureDetector_create"):
            logger.warning("OpenCV >= 2.4.3 required. Returning None.")
            return None
        track = surfTracker(self, bb, ts, **kwargs)
        ts.append(track)

    elif method.lower() == "mftrack":
        track = mfTracker(self, bb, ts, img, **kwargs)
        ts.append(track)

    return ts
