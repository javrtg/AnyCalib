from torch import Tensor

import anycalib.cameras as cams
from anycalib.cameras.base import BaseCamera


class CameraFactory:
    """Factory for camera models."""

    FACTORY: dict[str, type[BaseCamera]] = {
        "pinhole": cams.Pinhole,  # fx, fy, cx, cy
        "simple_pinhole": cams.SimplePinhole,  # f, cx, cy
        "radial": cams.Radial,  # fx, fy, cx, cy, k1, [k2, [k3, [k4]]]
        "simple_radial": cams.SimpleRadial,  # f, cx, cy, k1, [k2, [k3, [k4]]]
        "kb": cams.KannalaBrandt,  # fx, fy, cx, cy, k1, [k2, [k3, [k4]]]
        "simple_kb": cams.SimpleKannalaBrandt,  # f, cx, cy, k1, [k2, [k3, [k4]]]
        "ucm": cams.UCM,  # Unified Camera Model: fx, fy, cx, cy, ξ
        "simple_ucm": cams.SimpleUCM,  # f, cx, cy, ξ
        "eucm": cams.EUCM,  # Enhanced Unified Camera Model: fx, fy, cx, cy, 𝛼, β
        "simple_eucm": cams.SimpleEUCM,  # f, cx, cy, 𝛼, β
        "division": cams.Division,  # Division Model: fx, fy, cx, cy, k1, [k2, [k3, [k4]]]
        "simple_division": cams.SimpleDivision,  # f, cx, cy, k1, [k2, [k3, [k4]]]
        "fov": cams.FOV,  # Field of View Camera Model: fx, fy, cx, cy, ω
        # "div-odd": None,  # Division-odd
        # "ds": None,  # Double Sphere
    }

    __slots__ = ()

    @staticmethod
    def has(cam_id: str) -> bool:
        return cam_id in CameraFactory.FACTORY

    @staticmethod
    def get(cam_id: str) -> type[BaseCamera]:
        return CameraFactory.FACTORY[cam_id]

    @staticmethod
    def create_from_params(cam_id: str, params: Tensor) -> BaseCamera:
        return CameraFactory.FACTORY[cam_id].create_from_params(params)

    @staticmethod
    def create_from_id(cam_id: str) -> BaseCamera:
        """Create a camera model from an identifier of the form: {name}_{spec}

        Examples:
            "pinhole": Pinhole
            "simple_pinhole": SimplePinhole
            "radial_2": Radial with num_k=1
            "simple_radial_1": SimpleRadial with num_k=2
        """
        name = cam_id.partition(":")[0]
        return CameraFactory.FACTORY[name].create_from_id(cam_id)

    @staticmethod
    def create(cam_id: str, *args, **kwargs) -> BaseCamera:
        return CameraFactory.FACTORY[cam_id](*args, **kwargs)

    # @staticmethod
    # def project(cam_id: str, params: Tensor, points_3d: Tensor) -> Tensor:
    #     """Project 3D points in the reference of the camera to image coordinates

    #     Args:
    #         cam_id: Camera model.
    #         params: (..., D) Intrinsics parameters.
    #         points_3d: (..., N, 3) 3D points in the reference of the camera.

    #     Returns:
    #         (..., 2) image coordinates.
    #     """
    #     return CameraFactory.FACTORY[cam_id].project(params, points_3d)

    # @staticmethod
    # def unproject(cam_id: str, params: Tensor, points_2d: Tensor) -> Tensor:
    #     """Unproject image coordinates to unit bearing vectors in the camera frame.

    #     Args:
    #         cam_id: Camera model.
    #         params: (..., D) Intrinsic parameters.
    #         points_2d: (..., N, 2) image coordinates.

    #     Returns:
    #         (..., 3) unit bearing vectors in the camera frame.
    #     """
    #     return CameraFactory.FACTORY[cam_id].unproject(params, points_2d)

    # @staticmethod
    # def fit_minimal(
    #     cam_id: str, im_coords: Tensor, bearings: Tensor, order: int = -1
    # ) -> Tensor:
    #     """Minimal intrinsics estimation.

    #     Args:
    #         cam_id: Camera model to fit.
    #         im_coords: (..., MIN_SIZE, 2) image coordinates.
    #         points_3d: (..., MIN_SIZE, 3) 3D points in the reference of the camera.
    #         order: Order of the distortion model (ignored for pinhole). If negative,
    #             the order is set to the default value of the camera model.

    #     Returns:
    #         (..., D) fitted intrinsic parameters of the camera model.
    #     """
    #     return CameraFactory.FACTORY[cam_id].fit_minimal(im_coords, bearings, order)

    # @staticmethod
    # def fit_minimal_with_cxcy(
    #     cam_id: str, im_coords: Tensor, bearings: Tensor, cxcy: Tensor, order: int = -1
    # ):
    #     """Minimal intrinsics estimation with known principal point.

    #     Args:
    #         cam_id: Camera model to fit.
    #         im_coords: (..., MIN_SIZE, 2) image coordinates.
    #         points_3d: (..., MIN_SIZE, 3) 3D points in the reference of the camera.
    #         cxcy: (..., 2) known principal point.
    #         order: Order of the distortion model (ignored for pinhole). If negative,
    #             the order is set to the default value of the camera model.

    #     Returns:
    #         (..., D) fitted intrinsic parameters of the camera model.
    #     """
    #     return CameraFactory.FACTORY[cam_id].fit_minimal_with_cxcy(
    #         im_coords, bearings, cxcy, order
    #     )

    # @staticmethod
    # def fit(
    #     cam_id: str, im_coords: Tensor, bearings: Tensor, order: int = -1
    # ) -> Tensor:
    #     """Nonminimal intrinsics estimation.

    #     Args:
    #         cam_id: identifier of the camera model to fit.
    #         im_coords: (..., N, 2) image coordinates.
    #         bearings: (..., N, 3) unit bearing vectors in the camera frame.
    #         order: Order of the distortion model (ignored for pinhole). If negative,
    #             the order is set to the default value of the camera model.

    #     Returns:
    #         (..., D) fitted intrinsic parameters of the camera model.
    #     """
    #     return CameraFactory.FACTORY[cam_id].fit(im_coords, bearings, order)

    # @staticmethod
    # def fit_with_cxcy(
    #     cam_id: str, im_coords: Tensor, bearings: Tensor, cxcy: Tensor, order: int = -1
    # ) -> Tensor:
    #     """Nonminimal intrinsics estimation with known principal point.

    #     Args:
    #         cam_id: Camera model to fit.
    #         im_coords: (..., N, 2) image coordinates.
    #         bearings: (..., N, 3) unit bearing vectors in the camera frame.
    #         cxcy: (..., 2) known principal point.
    #         order: Order of the distortion model (ignored for pinhole). If negative,
    #             the order is set to the default value of the camera model.

    #     Returns:
    #         (..., D) fitted intrinsic parameters of the camera model.
    #     """
    #     return CameraFactory.FACTORY[cam_id].fit_with_cxcy(
    #         im_coords, bearings, cxcy, order
    #     )

    # @staticmethod
    # def fit_with_covs(
    #     cam_id: str,
    #     im_coords: Tensor,
    #     bearings: Tensor,
    #     covariances: Tensor,
    #     params0: Tensor | None,
    # ) -> Tensor:
    #     """Nonminimal intrinsics estimation using bearing covariances.

    #     Args:
    #         im_coords: (..., N, 2) image coordinates.
    #         bearings: (..., N, 3) unit bearing vectors in the camera frame.
    #         covariances: (..., N, 2) diagonal elements of the covariances
    #             expressed in the tangent space of the input bearings.
    #         params0: (..., 4) approximate estimation of the intrinsic parameters
    #             to propagate the covariances from the tangent space to the error
    #             space.

    #     Returns:
    #         (..., D) fitted intrinsic parameters of the camera model.
    #     """
    #     if params0 is None and not cam_id.startswith("pinhole"):
    #         raise ValueError("params0 must be provided for non-pinhole models.")
    #     return CameraFactory.FACTORY[cam_id].fit_with_covs(
    #         im_coords, bearings, covariances, params0=params0
    #     )

    # @staticmethod
    # def fit_with_covs_cxcy(
    #     cam_id: str,
    #     im_coords: Tensor,
    #     bearings: Tensor,
    #     covariances: Tensor,
    #     cxcy: Tensor,
    #     params0: Tensor | None,
    # ) -> Tensor:
    #     """Nonminimal intrinsics estimation using bearing covariances with known principal point.

    #     Args:
    #         im_coords: (..., N, 2) image coordinates.
    #         bearings: (..., N, 3) unit bearing vectors in the camera frame.
    #         covariances: (..., N, 2) diagonal elements of the covariances
    #             expressed in the tangent space of the input bearings.
    #         cxcy: (..., 2) known principal point.
    #         params0: (..., 4) approximate estimation of the intrinsic parameters
    #             to propagate the covariances from the tangent space to the error
    #             space.

    #     Returns:
    #         (..., D) fitted intrinsic parameters of the camera model.
    #     """
    #     if params0 is None and not cam_id.startswith("pinhole"):
    #         raise ValueError("params0 must be provided for non-pinhole models.")
    #     return CameraFactory.FACTORY[cam_id].fit_with_covs_cxcy(
    #         im_coords, bearings, covariances, cxcy, params0=params0
    #     )

    # @staticmethod
    # def map_params(
    #     cam_id_current: str,
    #     params_current: Tensor,
    #     cam_id_target: str,
    #     im_size: Tensor | None = None,
    # ) -> Tensor:
    #     """Fit a camera model from another model.

    #     Args:
    #         cam_id_current: Current camera model.
    #         params_current: (..., D_current) Intrinsic parameters of the current camera
    #             model.
    #         cam_id_target: Target camera model.
    #         im_size: (..., 2) required for tranformation between camera models with
    #             radial distortion. Ignored otherwise,

    #     Returns:
    #         (..., D_target) fitted intrinsic parameters of the target camera model.
    #     """
    #     return CameraFactory.FACTORY[cam_id_current].map_params(
    #         params_current, cam_id_target, im_size
    #     )
