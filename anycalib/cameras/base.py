from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseCamera(ABC):
    MAX_NPARAMS: int = 8

    PARAMS_IDX: dict[str, int]
    NAME: str
    NUM_F: int
    num_k = 0  # must be overridden with the number of distortion parameters

    @classmethod
    @abstractmethod
    def create_from_params(cls, params: Tensor) -> "BaseCamera":
        """Create a camera model from intrinsic parameters."""

    @classmethod
    @abstractmethod
    def create_from_id(cls, id_: str) -> "BaseCamera":
        """Create a camera model from an identifier."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the camera model."""

    @abstractmethod
    def validate_params(self, params: Tensor) -> bool:
        """Check if the parameters are valid based on the model."""

    @abstractmethod
    def get_min_sample_size(self, with_cxcy: bool) -> int:
        """Minimal number of 2D-3D samples needed to fit the intrinsics."""

    @abstractmethod
    def project(
        self, params: Tensor, points_3d: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Project 3D points in the reference of the camera to image coordinates.

        Args:
            params: (..., D) intrinsic parameters.
            points_3d: (..., N, 3) 3D points in the reference of each camera.

        Returns:
            (..., N, 2) image coordinates.
        """

    @abstractmethod
    def unproject(
        self, params: Tensor, points_2d: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Unproject image coordinates to unit bearing vectors in the camera frame.

        Args:
            params: (..., D) intrinsic parameters.
            points_2d: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3) unit bearing vectors in the camera frame.
            (..., N) boolean mask of valid unprojections. If None, all unprojections are
                valid.
        """

    @abstractmethod
    def fit_minimal(
        self, im_coords: Tensor, bearings: Tensor, cxcy: Tensor | None = None
    ) -> Tensor:
        """Fit instrinsics with a minimal set of 2D-bearing correspondences.

        Args:
            im_coords: (..., MIN_SAMPLE_SIZE, 2) image coordinates.
            bearings: (..., MIN_SAMPLE_SIZE, 3) unit bearing vectors.

        Returns:
            (..., D) intrinsic parameters.
        """

    @abstractmethod
    def fit(
        self,
        im_coords: Tensor,
        bearings: Tensor,
        cxcy: Tensor | None = None,
        covs: Tensor | None = None,
        params0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Fit instrinsics with a set of 2D-bearing correspondences.

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors.
            cxcy: (..., 2) known principal points.
            covs: (..., N, 2) diagonal elements of the covariances expressed in the
                tangent space of the input bearings.
            params0: (..., D) approximate estimation of the intrinsic parameters that
                may be needed (depending on the camera model) to propagate the
                covariances from the tangent space to the error space.

        Returns:
            (..., D) intrinsic parameters.
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found (<0) or the system is singular (>0).
        """

    @abstractmethod
    def jac_bearings_wrt_params(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Compute the Jacobian of the bearings wrt the intrinsic parameters.

        Args:
            params: (..., D) intrinsic parameters.
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, D) Jacobians.
        """

    @abstractmethod
    def jac_bearings_wrt_imcoords(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Compute the Jacobian of the bearings wrt the intrinsic parameters.

        Args:
            params: (..., D) intrinsic parameters.
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, 2) Jacobians.
        """

    def params_to_dict(self, params: Tensor) -> dict[str, Tensor]:
        """Convert intrinsic parameters to a dictionary.

        This method assumes the following standard parameter ordering:
            - focal length(s) (f or fx, fy)
            - principal point (cx, cy)
            - (optional) distortion parameters (k1, k2, ...)
        """
        self.validate_params(params)
        num_f = self.NUM_F
        return {
            "f": params[..., :num_f],
            "c": params[..., num_f : num_f + 2],
            "k": params[..., num_f + 2 :],  # empty tensor for models without distortion
        }

    def scale_and_shift(
        self, params: Tensor, scale: Tensor, shift: Tensor, copy: bool = True
    ):
        """Scale the focal length(s) and scale and shift the principal point.

        This method assumes that the origin (0, 0) of image coordinates is located at
        the *top-left* corner of the top-left pixel. Furthermore, this method assumes
        that the focal length(s) are the first parameter(s) of the intrinsic parameters,
        followed by the principal point. This method must be overridden if this is not
        true for a specific camera model.

        Args:
            params: (..., D) intrinsic parameters.
            scale: (..., 1) or (..., 2) scaling factors.
            shift: (..., 2) shift factors.
            copy: whether to return a new tensor or modify the input tensor in place.

        Returns:
            (..., D) intrinsics with scaled focal lengths and scale and shifted principal point.
        """
        self.validate_params(params)
        num_f = self.NUM_F
        if (
            num_f == 1
            and scale.shape[-1] == 2
            and ((scale[..., 0] - scale[..., 1]).abs() > 1e-2).any()
        ):
            raise ValueError(
                f"Different scales in x and y, but just one focal length: {scale}"
            )
        elif num_f == 1 and scale.shape[-1] == 2:
            scale = scale.mean(dim=-1, keepdim=True)
        scale = scale[..., 0, None] if num_f == 1 else scale

        params = params.clone() if copy else params
        params[..., :num_f] = params[..., :num_f] * scale
        params[..., num_f : num_f + 2] = params[..., num_f : num_f + 2] * scale + shift
        return params

    def reverse_scale_and_shift(
        self, params: Tensor, scale: Tensor, shift: Tensor, copy: bool = True
    ):
        """Reverse focal length(s) scaling and scale-shift of the principal point.

        This method assumes that the origin (0, 0) of image coordinates is located at
        the *top-left* corner of the top-left pixel. Furthermore, this method assumes
        that the focal length(s) are the first parameter(s) of the intrinsic parameters,
        followed by the principal point. This method must be overridden if this is not
        true for a specific camera model.

        Args:
            params: (..., D) intrinsic parameters.
            scale: (..., 1) or (..., 2) scaling factors.
            shift: (..., 2) shift factors.
            copy: whether to return a new tensor or modify the input tensor in place.

        Returns:
            (..., D) updated intrinsics.
        """
        self.validate_params(params)
        num_f = self.NUM_F
        if (
            num_f == 1
            and scale.shape[-1] == 2
            and ((scale[..., 0] - scale[..., 1]).abs() > 1e-2).any()
        ):
            raise ValueError(
                f"Different scales in x and y, but just one focal length: {scale}"
            )
        elif num_f == 1 and scale.shape[-1] == 2:
            scale = scale.mean(dim=-1, keepdim=True)
        scale = scale[..., 0, None] if num_f == 1 else scale

        params = params.clone() if copy else params
        params[..., :num_f] = params[..., :num_f] / scale
        params[..., num_f : num_f + 2] = (
            params[..., num_f : num_f + 2] - shift
        ) / scale
        return params

    @staticmethod
    def pixel_grid_coords(
        h: int, w: int, ref_tensor: Tensor, offset: float = 0.5
    ) -> Tensor:
        """Generate image coordinates for a given image size.

        The `offset` argument is useful for defining what does the grid represent:
        pixel centers, pixel corners, etc. For instance, in this work, we assume that
        the origin (0, 0) of image coordinates is located at the *top-left* corner of
        the top-left pixel. Therefore, the default offset of 0.5 implies that the
        default grid represents pixel centers.

        Args:
            h: image height.
            w: image width.
            ref_tensor: reference tensor to determine the device and dtype.
            offset: smallest coordinate value.

        Returns:
            (H, W, 2) image coordinates.
        """
        x = torch.arange(w, device=ref_tensor.device, dtype=ref_tensor.dtype)
        y = torch.arange(h, device=ref_tensor.device, dtype=ref_tensor.dtype)
        im_coords = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)
        return im_coords + offset

    def ray_grid(
        self, h: int, w: int, params: Tensor, offset: float = 0.5
    ) -> tuple[Tensor, Tensor | None]:
        """Generate a uniform grid of unit bearing vectors for a given image size.

        Args:
            h: image height.
            w: image width.
            params: (..., D) intrinsic parameters.
            offset: smallest image coordinate value.

        Returns:
            (H, W, 3) unit bearing vectors.
        """
        im_coords = self.pixel_grid_coords(h, w, params, offset)
        unit_bearings, valid = self.unproject(params, im_coords)
        return unit_bearings, valid

    def get_vfov(self, params: Tensor, h: Tensor | int) -> tuple[Tensor, Tensor]:
        """Model-agnostic way of computing the vertical field of view.

        This method assumes that the origin (0, 0) of image coordinates is located at
        the *top-left* corner of the top-left pixel.

        Args:
            params: (..., D) intrinsic parameters.
            h: (...,) image height in pixels. If int, it is assumed to be the
                same for all cameras.

        Returns:
            (...,) vfov: vertical field of view in radians.
            (...,) valid: boolean mask for flagging invalid vfovs (stemming from invalid
                unprojections).
        """
        self.validate_params(params)
        if isinstance(h, int):
            h = params.new_full(params.shape[:-1], h)
        assert h.shape == params.shape[:-1]
        # extreme vertical image coordinates: (cx, 0) and (cx, h)
        cx = params[..., self.PARAMS_IDX["cx"]]
        im_coords = params.new_zeros((*params.shape[:-1], 2, 2))
        im_coords[..., 0] = cx.unsqueeze(-1)
        im_coords[..., 1, 1] = h
        # unproject
        bearings, valid = self.unproject(params, im_coords)  # (..., 2, 3)
        valid = (
            h.new_ones(h.shape, dtype=torch.bool)
            if valid is None
            else valid.all(dim=-1)
        )
        # vfov = sum of angles between each ray and the optical axis ([0, 0, 1])
        angles = torch.acos(bearings[..., 2].clamp(-1, 1))  # (..., 2)
        vfov = angles.sum(dim=-1)
        return vfov, valid

    def get_hfov(self, params: Tensor, w: Tensor | int) -> tuple[Tensor, Tensor]:
        """Model-agnostic way of computing the horizontal field of view.

        This method assumes that the origin (0, 0) of image coordinates is located at
        the *top-left* corner of the top-left pixel.

        Args:
            params: (..., D) intrinsic parameters.
            w: (...,) image width in pixels. If int, it is assumed to be the
                same for all cameras.

        Returns:
            (...,) hfov: horizontal field of view in radians.
            (...,) valid: boolean mask for flagging invalid hfovs (stemming from invalid
                unprojections).
        """
        self.validate_params(params)
        if isinstance(w, int):
            w = params.new_full(params.shape[:-1], w)
        assert w.shape == params.shape[:-1]
        # extreme horizontal image coordinates: (0, cy) and (w, cy)
        cy = params[..., self.PARAMS_IDX["cy"]]
        im_coords = params.new_zeros((*params.shape[:-1], 2, 2))
        im_coords[..., 1] = cy.unsqueeze(-1)
        im_coords[..., 1, 0] = w
        # unproject
        bearings, valid = self.unproject(params, im_coords)  # (..., 2, 3)
        valid = (
            w.new_ones(w.shape, dtype=torch.bool)
            if valid is None
            else valid.all(dim=-1)
        )
        # hfov = sum of angles between each ray and the optical axis ([0, 0, 1])
        angles = torch.acos(bearings[..., 2].clamp(-1, 1))  # (..., 2)
        hfov = angles.sum(dim=-1)
        return hfov, valid

    def get_pinhole_vfov(self, params: Tensor, h: Tensor | int) -> Tensor:
        """Compute the vertical field of view with the analytical formula corresponding
        to a pinhole camera with principal point at the image center.

        Args:
            params: (..., D) intrinsic parameters.
            h: (...,) image height in pixels. If int, it is assumed to be the
                same for all cameras.

        Returns:
            (...,) vertical field of view in radians.
        """
        assert "f" in self.PARAMS_IDX or "fy" in self.PARAMS_IDX, (
            f"focal length not found in camera model {self.NAME}"
        )
        self.validate_params(params)

        if isinstance(h, int):
            h = params.new_full(params.shape[:-1], h)
        assert h.shape == params.shape[:-1]

        fy = params[..., self.PARAMS_IDX.get("f", self.PARAMS_IDX.get("fy"))]
        return 2 * torch.atan2(0.5 * h, fy)

    def get_optim_update(self, params: Tensor, delta: Tensor) -> Tensor:
        """If needed, undo reparameterization in delta and apply the update.

        During optimization, some cameras may have parameters with a constrained domain,
        such as the parameter ξ (>0) of the UCM camera model. Because of that, during
        the optimization these parameters are reparameterized so that this domain is not
        violated. This method should be overridden by camera models that need to undo
        this reparameterization before applying the update.

        Args:
            params: (..., D) intrinsic parameters.
            delta: (..., D) update.

        Returns:
            (..., D) updated intrinsic parameters.
        """
        return params + delta

    def get_optim_jac(self, jac: Tensor, params: Tensor) -> Tensor:
        """If needed, apply chain rule for reparameterized parameters.

        Args:
            jac: (..., N, D) Jacobian.
            params: (..., D) intrinsic parameters.

        Returns:
            (..., N, D) updated Jacobian.
        """
        return jac

    def fit_dist_from_radii(
        self, r: Tensor, R: Tensor, Z: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Fit distortion parameters from sensor and ray radii.

        Args:
            r: (..., N) radii (from center of distortion) in the sensor plane.
            R: (..., N) radii of the rays: R = sqrt(X^2 + Y^2)
            Z: (..., N) Z coordinates of the rays.

        Returns:
            (..., D_k) distortion parameters.
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found and/or the system is singular. None if all
                estimates are valid.
        """
        raise NotImplementedError

    @property
    def nparams(self) -> int:
        nparams = 2  # principal point
        if hasattr(self, "NUM_F"):
            nparams += self.NUM_F
        if hasattr(self, "num_k"):
            nparams += self.num_k  # type: ignore
        return nparams

    def undistort_image(
        self,
        im: Tensor,
        params: Tensor,
        scale: float = 1.0,
        target_proj: str = "perspective",
        outside_value: float = 1.0,
        interp_mode: str = "bilinear",
    ) -> Tensor:
        """Undistort an image using the distortion model.

        Assumptions of this method:
        1.  The focal length(s) follow the general convention of being the first (1, 2)
            parameter(s) of the given Tensor, followed by the principal point (cx, cy).
        2.  This method also assumes that the focal length is expressed in pixels, and
        3.  that the principal point coordinates are expressed w.r.t. an image plane
            coordinate system whose origin (0, 0) is the top-left corner of the top-left pixel.

        Args:
            im: (B, 3, H, W) or (3, H, W) image.
            params: (B, D) intrinsic parameters.
            scale: scaling factor for the focal length(s).
            target_proj: target projection model for the undistortion. See options in
                the method `ideal_unprojection`.
            outside_value: value to use for pixels outside the image bounds after
                undistortion.
            interp_mode: interpolation mode for the grid sample operation. Default is
                "bilinear". Other options are "nearest" and "bicubic".

        Returns:
            (B, 3, H, W) or (3, H, W) undistorted image.
        """
        assert im.ndim in (3, 4), f"Expected 3 or 4 input dimensions, got {im.ndim=}."
        is_batched = im.ndim == 4
        if not is_batched:
            im = im[None]
        assert scale > 0, f"scale must be positive, got {scale=}"
        b, _, h, w = im.shape
        num_f = self.NUM_F
        f, c = params[..., None, :num_f], params[..., None, num_f : num_f + 2]
        # normalized image coordinates
        im_n = (self.pixel_grid_coords(h, w, params, 0.0).reshape(-1, 2) - c) / f
        r = torch.linalg.norm(im_n, dim=-1) / scale  # (B, H*W)
        # get forward distortion map
        theta = self.ideal_unprojection(r, target_proj)
        phi = torch.atan2(im_n[..., 1], im_n[..., 0])
        R = torch.sin(theta)
        rays = torch.stack(
            (R * torch.cos(phi), R * torch.sin(phi), torch.cos(theta)), dim=-1
        )  # (B, H*W, 3)
        if num_f == 2:
            params = params.clone()
            params[..., :2] = f.amax(dim=-1, keepdim=True)
        map_xy, valid = self.project(params, rays)
        if valid is not None and not valid.all():
            print(f"Warning: {~valid.sum()} invalid projections.")
        # normalize coords to [-1, 1]
        map_xy = 2 * map_xy.reshape(b, h, w, 2) / map_xy.new_tensor((w, h)) - 1
        # undistort
        im_undist = outside_value + torch.nn.functional.grid_sample(
            im - outside_value,
            map_xy,
            mode=interp_mode,
            padding_mode="zeros",
            align_corners=False,
        )
        if not is_batched:
            im_undist = im_undist[0]
        return im_undist

    @staticmethod
    def ideal_unprojection(r: Tensor, target_proj: str) -> Tensor:
        """Compute the ideal (radial) target unprojection

        Args:
            r: (..., N) radii of *normalized* image coordinates.
            target_proj: target projection model.

        Returns:
            (..., N) ideal unprojection.
        """
        assert target_proj in {
            "perspective",
            "rectilinear",
            "equisolid",
            "equidistant",
            "stereographic",
            "orthographic",
        }, f"Unknown projection model {target_proj}"

        if target_proj == "perspective" or target_proj == "rectilinear":
            theta = torch.arctan(r.clamp(0))
        elif target_proj == "equisolid":
            theta = 2 * torch.arcsin((0.5 * r).clamp(0, 1))
        elif target_proj == "equidistant":  # equidistant
            theta = r.clamp(0)
        elif target_proj == "stereographic":
            theta = 2 * torch.arctan((0.5 * r).clamp(0))
        elif target_proj == "orthographic":
            theta = torch.arcsin((r).clamp(0, 1))
        else:
            raise ValueError(f"Unknown projection: {target_proj}")
        return theta
