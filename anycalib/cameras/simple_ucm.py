from anycalib.cameras import UCM


class SimpleUCM(UCM):
    """Implementation of the Unified Camera Model (UCM) [1, Sec. II], with one focal length.

    The (ordered) intrinsic parameters are f, cx, cy, xi:
        - f [pixels] is the focal length,
        - (cx, cy) [pixels] is the principal points.
        - xi represents the distance from the center of projection to the center of the
            sphere and controls the magnitude of radial distortion present in the image.


    [1] Single View Point Omnidirectional Camera Calibration from Planar Grids.
        C Mei, P Rives, ICRA 2007.
    """

    NAME = "simple_ucm"
    # number of focal lengths
    NUM_F = 1
    PARAMS_IDX = {
        "f": 0,
        "cx": 1,
        "cy": 2,
        "k1": 3,  # xi
    }
    num_k = 1
