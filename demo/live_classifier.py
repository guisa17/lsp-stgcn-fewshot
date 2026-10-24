"""Live webcam classification with a trained run (qualitative check; nothing is saved to disk).

Keeps a rolling buffer of the last ``SEQ_LENGTH`` frames and, every ``--interval`` seconds once the
buffer is full, classifies it and draws the most probable classes.

Usage:
    python demo/live_classifier.py --runs-dir outputs/lsp --run lsp_hagcn
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np

from lsp import inference
from lsp.data.recording import extract_points
from lsp.utils.ipc import should_stop, write_status
from lsp.viz import hud

SEQ_LENGTH = 30
FRAME_SIZE = (640, 480)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5


def _draw_predictions(cv2, image, ranked, top_k) -> None:
    h, w = image.shape[:2]
    n = min(top_k, len(ranked))
    panel_h = 22 + n * 26
    hud.panel(cv2, image, 0, h - panel_h, w, h, alpha=0.5)
    bar_x0, bar_x1 = 130, w - 60
    for i in range(n):
        label, prob = ranked[i]
        y = h - panel_h + 22 + i * 26
        cv2.putText(
            image, str(label), (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud.TEXT, 1, cv2.LINE_AA
        )
        hud.progress_bar(
            cv2,
            image,
            bar_x0,
            y - 12,
            bar_x1,
            y - 4,
            prob,
            color=hud.ACCENT if i == 0 else hud.MUTED,
        )
        cv2.putText(
            image,
            f"{prob:.0%}",
            (bar_x1 + 6, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            hud.TEXT,
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--run", required=True, help="Run name under --runs-dir, e.g. lsp_hagcn")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--interval", type=float, default=0.6, help="Seconds between predictions")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--status-file", default=None)
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()

    write_status(args.status_file, state="loading", run=args.run)
    try:
        run = inference.load_run(args.runs_dir, args.run)
    except Exception as e:
        write_status(args.status_file, state="error", error=str(e))
        raise SystemExit(str(e)) from e

    import cv2
    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    pose_spec, hand_spec = hud.landmark_specs(mp_drawing)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    if not cap.isOpened():
        write_status(args.status_file, state="error", error=f"Could not open camera {args.camera}")
        raise SystemExit(f"Could not open camera {args.camera}")

    buffer: deque = deque(maxlen=SEQ_LENGTH)
    last_ranked: list = []
    last_pred_ts = 0.0
    window = f"Live classification: {args.run}"
    write_status(args.status_file, state="ready", run=args.run, classes=run.classes)

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            refine_face_landmarks=True,
            model_complexity=args.model_complexity,
        ) as holistic:
            while not should_stop(args.stop_file):
                ret, frame = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                for lm, conn, spec in (
                    (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_spec),
                    (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_spec),
                    (results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, pose_spec),
                ):
                    mp_drawing.draw_landmarks(image, lm, conn, spec, spec)
                buffer.append(extract_points(results))

                h, w = image.shape[:2]
                hud.panel(cv2, image, 0, 0, w, 40)
                cv2.putText(
                    image,
                    args.run,
                    (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    hud.TEXT,
                    1,
                    cv2.LINE_AA,
                )
                now = time.time()
                if len(buffer) < SEQ_LENGTH:
                    hud.progress_bar(cv2, image, 160, 16, w - 18, 24, len(buffer) / SEQ_LENGTH)
                    cv2.putText(
                        image,
                        "filling buffer...",
                        (160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        hud.MUTED,
                        1,
                        cv2.LINE_AA,
                    )
                elif now - last_pred_ts >= args.interval:
                    result = run.predict_raw(np.array(buffer, dtype=np.float32))
                    last_ranked = result["ranked"][: args.top_k]
                    last_pred_ts = now
                    write_status(
                        args.status_file,
                        state="predicting",
                        run=args.run,
                        pred=result["pred"],
                        confidence=result["confidence"],
                        ranked=last_ranked,
                    )
                if last_ranked:
                    _draw_predictions(cv2, image, last_ranked, args.top_k)
                cv2.imshow(window, image)
                cv2.waitKey(1)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    write_status(
        args.status_file, state="stopped" if should_stop(args.stop_file) else "closed", run=args.run
    )


if __name__ == "__main__":
    main()
