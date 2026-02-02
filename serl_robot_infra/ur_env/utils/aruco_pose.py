from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import cv2


@dataclass
class ArucoDetection:
    marker_id: int
    corners: np.ndarray   # (4,2)
    rvec: np.ndarray      # (3,)
    tvec: np.ndarray      # (3,) meters
    area_px2: float


def _poly_area(corners_4x2: np.ndarray) -> float:
    c = corners_4x2.reshape(4, 2)
    x = c[:, 0]
    y = c[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


class ArucoPoseEstimator:
    def __init__(
        self,
        marker_length_m: float,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        *,
        dict_id: int = cv2.aruco.DICT_4X4_50,
        target_ids: Optional[Sequence[int]] = None,
    ):
        self.marker_length_m = float(marker_length_m)
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
        self.target_ids = set(target_ids) if target_ids is not None else None

        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        else:
            self.aruco_dict = cv2.aruco.Dictionary_get(dict_id)

        if hasattr(cv2.aruco, "DetectorParameters"):
            self.params = cv2.aruco.DetectorParameters()
        else:
            self.params = cv2.aruco.DetectorParameters_create()

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.params) if hasattr(cv2.aruco, "ArucoDetector") else None

    def estimatePoseSingleMarkers(self, corners, marker_size, mtx, distortion):
        '''
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        '''
        marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, marker_size / 2, 0],
                                [marker_size / 2, -marker_size / 2, 0],
                                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
        trash = []
        rvecs = []
        tvecs = []
        for c in corners:
            nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
            rvecs.append(R)
            tvecs.append(t)
            trash.append(nada)
        return np.asarray(rvecs), np.asarray(tvecs), np.asarray(trash)

    def detect(self, frame: np.ndarray) -> Optional[ArucoDetection]:
        if frame is None or frame.ndim != 3:
            return None

        # robust grayscale conversion independent-ish of channel order
        gray = frame.mean(axis=-1).astype(np.uint8)

        if self.detector is not None:
            corners_list, ids, _rej = self.detector.detectMarkers(gray)
        else:
            corners_list, ids, _rej = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        if ids is None or len(ids) == 0:
            return None

        ids = ids.flatten().astype(int)

        candidates = []
        for i, marker_id in enumerate(ids):
            if self.target_ids is not None and marker_id not in self.target_ids:
                continue
            corners = np.asarray(corners_list[i], dtype=np.float64).reshape(4, 2)
            area = _poly_area(corners)
            candidates.append((area, marker_id, corners))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        area, marker_id, corners = candidates[0]

        rvecs, tvecs, _obj = self.estimatePoseSingleMarkers(
            [corners.astype(np.float32)],
            self.marker_length_m,
            self.camera_matrix,
            self.dist_coeffs,
        )
        rvec = np.asarray(rvecs[0], dtype=np.float64).reshape(3)
        tvec = np.asarray(tvecs[0], dtype=np.float64).reshape(3)

        return ArucoDetection(
            marker_id=int(marker_id),
            corners=corners,
            rvec=rvec,
            tvec=tvec,
            area_px2=float(area),
        )
