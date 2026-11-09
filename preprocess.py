"""Image preparation before OCR.

Fixed order: perspective -> grayscale -> denoise -> contrast -> deskew.
No hard binarization: EasyOCR reads gray photo pages better than black/white.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

MEDIAN_BLUR_SIZE = 3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)

DETECT_MAX_SIDE = 1000
MIN_DOCUMENT_AREA_RATIO = 0.30
MAX_DOCUMENT_AREA_RATIO = 0.95
APPROX_EPSILON_RATIO = 0.02
MIN_SHEET_SIDE = 100
MIN_SIDE_TO_LONG_SIDE = 0.20
BORDER_MARGIN_RATIO = 0.02

MIN_SKEW_DEGREES = 0.4
MAX_SKEW_DEGREES = 15.0
MIN_TEXT_PIXELS = 100


def preprocess(image: np.ndarray) -> np.ndarray:
    """Prepare a photo for OCR and return a single-channel image."""
    aligned = straighten_document(image)
    gray = to_gray(aligned)
    denoised = cv2.medianBlur(gray, MEDIAN_BLUR_SIZE)
    contrasted = increase_contrast(denoised)
    return deskew(contrasted)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def increase_contrast(gray: np.ndarray) -> np.ndarray:
    """Raise local contrast. Tiny frames are returned unchanged."""
    height, width = gray.shape[:2]
    tile_width, tile_height = CLAHE_TILE_SIZE
    if height < tile_height or width < tile_width:
        return gray

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    return clahe.apply(gray)


def straighten_document(image: np.ndarray) -> np.ndarray:
    """Correct perspective only when the contour looks like a separate sheet."""
    corners = find_document_corners(image)
    if corners is None:
        return image
    return warp_document(image, corners)


def find_document_corners(image: np.ndarray) -> Optional[np.ndarray]:
    """Find four sheet corners. None means perspective warp is skipped."""
    small, scale = resize_for_detection(image)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    largest = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    image_area = float(small.shape[0] * small.shape[1])

    for contour in largest:
        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, APPROX_EPSILON_RATIO * perimeter, closed=True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        area_ratio = cv2.contourArea(approx) / image_area
        if not MIN_DOCUMENT_AREA_RATIO <= area_ratio <= MAX_DOCUMENT_AREA_RATIO:
            continue

        points = approx.reshape(4, 2).astype(np.float32) / scale
        ordered = order_corners(points)
        if accept_document_corners(ordered, image.shape):
            return ordered
    return None


def resize_for_detection(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale for contour search. scale maps points back to the original size."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= DETECT_MAX_SIDE:
        return image, 1.0

    scale = DETECT_MAX_SIDE / float(longest)
    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def order_corners(points: np.ndarray) -> np.ndarray:
    """Corner order: top-left, top-right, bottom-right, bottom-left."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    difference = points[:, 1] - points[:, 0]
    ordered[1] = points[np.argmin(difference)]
    ordered[3] = points[np.argmax(difference)]
    return ordered


def accept_document_corners(corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    """Reject skinny contours and a contour that is the frame itself."""
    sides = corner_side_lengths(corners)
    longest = max(sides)
    if longest < MIN_SHEET_SIDE:
        return False
    if min(sides) < MIN_SIDE_TO_LONG_SIDE * longest:
        return False
    return corners_are_inside_frame(corners, image_shape)


def corner_side_lengths(corners: np.ndarray) -> list[float]:
    pairs = (
        (corners[0], corners[1]),
        (corners[1], corners[2]),
        (corners[2], corners[3]),
        (corners[3], corners[0]),
    )
    return [float(np.linalg.norm(end - start)) for start, end in pairs]


def corners_are_inside_frame(corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    """If every corner sits on the frame border, this is the frame — do not warp."""
    height, width = image_shape[:2]
    margin_x = width * BORDER_MARGIN_RATIO
    margin_y = height * BORDER_MARGIN_RATIO
    corners_on_border = 0

    for x, y in corners:
        on_border = (
            x <= margin_x
            or y <= margin_y
            or x >= width - 1 - margin_x
            or y >= height - 1 - margin_y
        )
        if on_border:
            corners_on_border += 1

    return corners_on_border < 4


def warp_document(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Map the detected quadrilateral onto a rectangular page."""
    top = float(np.linalg.norm(corners[1] - corners[0]))
    bottom = float(np.linalg.norm(corners[2] - corners[3]))
    left = float(np.linalg.norm(corners[3] - corners[0]))
    right = float(np.linalg.norm(corners[2] - corners[1]))
    width = int(round(max(top, bottom)))
    height = int(round(max(left, right)))
    if width < MIN_SHEET_SIDE or height < MIN_SHEET_SIDE:
        return image

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate text toward horizontal. Uncertain angles are left unchanged."""
    angle = estimate_skew_degrees(gray)
    if angle is None:
        return gray

    height, width = gray.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def estimate_skew_degrees(gray: np.ndarray) -> Optional[float]:
    """Estimate skew from dark text pixels. None means rotation must not run."""
    _threshold, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    points = cv2.findNonZero(binary)
    if points is None or len(points) < MIN_TEXT_PIXELS:
        return None

    _center, _size, raw_angle = cv2.minAreaRect(points)
    angle = rotation_to_level_text(float(raw_angle))
    if abs(angle) < MIN_SKEW_DEGREES or abs(angle) > MAX_SKEW_DEGREES:
        return None
    return angle


def rotation_to_level_text(raw_angle: float) -> float:
    """Convert minAreaRect angle into a cv2.getRotationMatrix2D angle."""
    if raw_angle < -45.0:
        return -(90.0 + raw_angle)
    return -raw_angle
