from anycalib.cameras import Division


class SimpleDivision(Division):
    """Implementation of the Division Camera Model [1] with one focal length.

    This class implements the slight variation [2, 3] of the original model [1] which
    defines the back-projection (or unprojection) function as:
        x = (u - cx)/f
        y = (v - cy)/f
        z =  1 + k1*r^2 + k2*r^4 + ...
    where r is the radius of the retinal point, defined as: r = sqrt(x^2 + y^2). The
    unprojected point is subsequently normalized to have unit norm. This implementation
    supports a variable number (up to 4) of distortion coefficients, controlled by the
    variable/attribute num_k.
    The (ordered) intrinsic parameters are f, cx, cy, k1, k2, ...
        - f [pixels] is the focal length,
        - (cx, cy) [pixels] is the principal points.
        - (k1, k2, ...) are the radial distortion coefficients.

    [1] Simultaneous Linear Estimation of Multiple View Geometry and Lens Distortion.
        A.W. Fitzgibbon, CVPR 2001.
    [2] Revisiting Radial Distortion Absolute Pose. V. Larsson et al., ICCV 2019.
    [3] Babelcalib: A Universal Approach to Calibrating Central Cameras.
        Y. Lochman et al., ICCV 2021.
    """

    NAME = "simple_division"
    # number of focal lengths
    NUM_F = 1
    PARAMS_IDX = {
        "f": 0,
        "cx": 1,
        "cy": 2,
        "k1": 3,
        "k2": 4,
        "k3": 5,
        "k4": 6,
    }
