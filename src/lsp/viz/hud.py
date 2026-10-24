"""Palette and drawing helpers for the live OpenCV windows (recorder and live classifier).

A muted, low-contrast palette so the overlay does not compete with the camera image.
"""

from __future__ import annotations

# BGR colours (OpenCV).
BG = (36, 32, 30)
TEXT = (228, 226, 222)
MUTED = (134, 130, 126)
ACCENT = (176, 148, 120)
RECORD = (122, 118, 192)
GOOD = (148, 172, 140)
WARN = (118, 172, 212)
TRACK = (70, 68, 66)

PANEL_ALPHA = 0.42


def panel(cv2, image, x0, y0, x1, y1, color=BG, alpha=PANEL_ALPHA) -> None:
    """Semi-transparent rectangle used as a background for overlay text."""
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def progress_bar(cv2, image, x0, y0, x1, y1, frac, color=ACCENT, track=TRACK) -> None:
    cv2.rectangle(image, (x0, y0), (x1, y1), track, 1, cv2.LINE_AA)
    filled = x0 + int((x1 - x0) * max(0.0, min(frac, 1.0)))
    if filled > x0:
        cv2.rectangle(image, (x0, y0), (filled, y1), color, -1)


def landmark_specs(mp_drawing):
    """(pose_spec, hand_spec) drawing specs using the muted palette and thin strokes."""
    pose_spec = mp_drawing.DrawingSpec(color=ACCENT, thickness=1, circle_radius=2)
    hand_spec = mp_drawing.DrawingSpec(color=WARN, thickness=1, circle_radius=2)
    return pose_spec, hand_spec
