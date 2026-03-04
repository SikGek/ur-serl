"""Camera helpers for the G1 HIL-SERL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraSpec:
    """Configuration for a single image source."""

    backend: str
    width: int = 1280
    height: int = 720
    fps: int = 30

    # OpenCV backend
    device: Optional[int] = None
    path: Optional[str] = None

    # RealSense backend
    serial_number: Optional[str] = None
    exposure: Optional[int] = None

    # Shared / post-processing
    crop: Optional[Tuple[int, int, int, int]] = None  # (y0, y1, x0, x1)


class _BaseCamera:
    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class _OpenCVCamera(_BaseCamera):
    def __init__(self, spec: CameraSpec) -> None:
        source = spec.path if spec.path is not None else int(spec.device or 0)
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
        self._cap.set(cv2.CAP_PROP_FPS, spec.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source {source!r}")

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("OpenCV camera read failed.")
        return frame

    def close(self) -> None:
        self._cap.release()


class _RealSenseCamera(_BaseCamera):
    def __init__(self, spec: CameraSpec) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:  # pragma: no cover - depends on user environment.
            raise ImportError(
                "pyrealsense2 is required for RealSense cameras. Install it or switch the "
                "camera backend to 'opencv'."
            ) from exc

        self._rs = rs
        self._pipeline = rs.pipeline()
        self._config = rs.config()

        if spec.serial_number:
            self._config.enable_device(spec.serial_number)

        self._config.enable_stream(
            rs.stream.color,
            spec.width,
            spec.height,
            rs.format.bgr8,
            spec.fps,
        )

        self._profile = self._pipeline.start(self._config)

        if spec.exposure is not None:
            sensor = self._profile.get_device().first_color_sensor()
            sensor.set_option(rs.option.enable_auto_exposure, 0)
            sensor.set_option(rs.option.exposure, int(spec.exposure))

    def read(self) -> np.ndarray:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense returned an empty color frame.")
        return np.asanyarray(color.get_data())

    def close(self) -> None:
        self._pipeline.stop()


class G1CameraManager:
    """Thin camera multiplexer used by the environment.

    The return convention mirrors HIL-SERL's Franka env:
        - ``images`` contains 128x128 RGB arrays for policy / classifier input.
        - ``full_res`` contains cropped full-resolution arrays for optional video saving.
    """

    def __init__(
        self,
        camera_cfg: Mapping[str, Mapping],
        *,
        output_hw: Tuple[int, int] = (128, 128),
    ) -> None:
        self._output_hw = tuple(int(x) for x in output_hw)
        self._cameras: Dict[str, _BaseCamera] = {}
        self._specs: Dict[str, CameraSpec] = {}

        for name, cfg in camera_cfg.items():
            spec = self._parse_spec(cfg)
            self._specs[name] = spec

            backend = spec.backend.lower()
            if backend == "opencv":
                self._cameras[name] = _OpenCVCamera(spec)
            elif backend == "realsense":
                self._cameras[name] = _RealSenseCamera(spec)
            else:
                raise ValueError(
                    f"Unsupported camera backend={spec.backend!r} for camera {name!r}."
                )

    @staticmethod
    def _parse_spec(cfg: Mapping) -> CameraSpec:
        crop = cfg.get("crop")
        if crop is not None:
            crop = tuple(int(v) for v in crop)
            if len(crop) != 4:
                raise ValueError("Camera crop must be [y0, y1, x0, x1].")

        return CameraSpec(
            backend=str(cfg.get("backend", "opencv")),
            width=int(cfg.get("width", 1280)),
            height=int(cfg.get("height", 720)),
            fps=int(cfg.get("fps", 30)),
            device=cfg.get("device"),
            path=cfg.get("path"),
            serial_number=cfg.get("serial_number"),
            exposure=cfg.get("exposure"),
            crop=crop,
        )

    def read(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Read every configured camera once."""
        images: Dict[str, np.ndarray] = {}
        full_res: Dict[str, np.ndarray] = {}

        for name, cam in self._cameras.items():
            frame_bgr = cam.read()
            frame_bgr = self._apply_crop(frame_bgr, self._specs[name].crop)
            full_res[name] = frame_bgr.copy()

            resized = cv2.resize(frame_bgr, self._output_hw[::-1])
            images[name] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return images, full_res

    @staticmethod
    def _apply_crop(frame: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        if crop is None:
            return frame
        y0, y1, x0, x1 = crop
        return frame[y0:y1, x0:x1]

    def close(self) -> None:
        for cam in self._cameras.values():
            cam.close()
