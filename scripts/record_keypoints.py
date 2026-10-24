"""Record sign takes from a webcam as MediaPipe Holistic keypoints (73-node layout).

Each take is ``SEQ_LENGTH`` frames of 95 raw points (pose 33 + left hand 21 + right hand 21 +
20 face points selected with ``FACE_IDX``), saved as ``<out>/<signer_id>/<CLASS>_<timestamp>.npy``
with shape ``(30, 285)``. No video or image is stored.

Recorded keypoints include facial landmarks and must be treated as personal data: ``--out`` must
point outside this repository, and recordings require the signer's informed consent.

Usage:
    pip install -e ".[demo]"
    python scripts/record_keypoints.py --out /path/outside/repo/raw --signer-id signer_00 \
        --classes HOLA,GRACIAS --n-samples 15

The demo app launches this script as a subprocess and passes ``--status-file``/``--stop-file``.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from lsp.data import canonical as C
from lsp.data.recording import detection_rates, extract_points, is_low_quality
from lsp.utils.ipc import should_stop, write_status
from lsp.viz import hud

SEQ_LENGTH = 30  # frames per take; the whole pipeline assumes T = 30
COUNTDOWN_S = 3  # seconds of countdown before each take
REST_FRAMES = 20  # frames shown between takes ...
REST_FRAME_MS = 100  # ... at this interval
FRAME_SIZE = (640, 480)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
WINDOW = "Recorder (73 nodes)"
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _draw_hud(
    cv2,
    image,
    *,
    word,
    sample_idx,
    n_samples,
    mode,
    frame_num=None,
    countdown=None,
    live_dots=None,
    last_saved=None,
) -> None:
    """Muted overlay: header with class and progress plus a state indicator."""
    h, w = image.shape[:2]
    hud.panel(cv2, image, 0, 0, w, 58)
    cv2.putText(image, word, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, hud.TEXT, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        f"take {sample_idx + 1}/{n_samples}",
        (18, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        hud.MUTED,
        1,
        cv2.LINE_AA,
    )
    hud.progress_bar(cv2, image, w - 190, 22, w - 18, 30, sample_idx / max(n_samples, 1))

    if mode == "countdown":
        cv2.circle(image, (w // 2, h // 2), 60, hud.WARN, 3, cv2.LINE_AA)
        cv2.putText(
            image,
            str(countdown),
            (w // 2 - 18, h // 2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            hud.WARN,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "get ready",
            (w // 2 - 62, h // 2 + 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            hud.MUTED,
            1,
            cv2.LINE_AA,
        )
    elif mode == "recording":
        cv2.circle(image, (24, h - 24), 7, hud.RECORD, -1, cv2.LINE_AA)
        cv2.putText(
            image, "REC", (40, h - 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hud.RECORD, 1, cv2.LINE_AA
        )
        hud.progress_bar(
            cv2, image, 96, h - 30, w - 18, h - 22, (frame_num + 1) / SEQ_LENGTH, color=hud.RECORD
        )
        if live_dots:
            lx = 18
            for ok in (live_dots.get("pose"), live_dots.get("lh"), live_dots.get("rh")):
                cv2.circle(image, (lx, 18), 4, hud.GOOD if ok else hud.MUTED, -1, cv2.LINE_AA)
                lx += 24
    elif mode == "rest":
        hud.panel(cv2, image, 0, h - 40, w, h)
        cv2.putText(
            image, "rest", (18, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hud.GOOD, 1, cv2.LINE_AA
        )

    if last_saved:
        r = last_saved["rates"]
        msg = f"saved -- pose {r['pose']:.0%}  left {r['lh']:.0%}  right {r['rh']:.0%}"
        color = hud.WARN if last_saved.get("low_quality") else hud.GOOD
        cv2.putText(
            image,
            msg,
            (18, h - 50 if mode != "rest" else h - 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def record_class(
    holistic,
    cap,
    cv2,
    mp_holistic,
    mp_drawing,
    word,
    n_samples,
    out_dir,
    status_file=None,
    stop_file=None,
    classes_done=None,
    classes_pending=None,
) -> None:
    """Record ``n_samples`` takes of one class with countdown, recording and rest phases."""
    os.makedirs(out_dir, exist_ok=True)
    last_saved = None
    pose_spec, hand_spec = hud.landmark_specs(mp_drawing)
    progress = {
        "clase": word,
        "n_samples": n_samples,
        "classes_done": classes_done,
        "classes_pending": classes_pending,
    }

    for sample_idx in range(n_samples):
        if should_stop(stop_file):
            write_status(status_file, state="stopped", sample_idx=sample_idx, **progress)
            return
        print(f"\n[{word}] take {sample_idx + 1}/{n_samples}")

        for i in range(COUNTDOWN_S, 0, -1):
            ret, frame = cap.read()
            if not ret:
                break
            _draw_hud(
                cv2,
                frame,
                word=word,
                sample_idx=sample_idx,
                n_samples=n_samples,
                mode="countdown",
                countdown=i,
                last_saved=last_saved,
            )
            cv2.imshow(WINDOW, frame)
            cv2.waitKey(1)
            write_status(
                status_file, state="countdown", sample_idx=sample_idx + 1, countdown=i, **progress
            )
            time.sleep(1.0)

        sequence = []
        for frame_num in range(SEQ_LENGTH):
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

            sequence.append(extract_points(results))  # (285,)
            live_dots = {
                "pose": results.pose_landmarks is not None,
                "lh": results.left_hand_landmarks is not None,
                "rh": results.right_hand_landmarks is not None,
            }
            _draw_hud(
                cv2,
                image,
                word=word,
                sample_idx=sample_idx,
                n_samples=n_samples,
                mode="recording",
                frame_num=frame_num,
                live_dots=live_dots,
                last_saved=last_saved,
            )
            cv2.imshow(WINDOW, image)
            cv2.waitKey(1)
            write_status(
                status_file,
                state="recording",
                sample_idx=sample_idx + 1,
                frame_num=frame_num + 1,
                seq_length=SEQ_LENGTH,
                **progress,
            )

        seq = np.array(sequence, dtype=np.float32)
        if seq.shape != (SEQ_LENGTH, C.RAW_FEATURES):
            print(f"  ! take discarded (shape {seq.shape})")
            write_status(status_file, state="discarded", sample_idx=sample_idx + 1, **progress)
            continue
        fname = f"{word}_{int(time.time() * 1000)}.npy"
        np.save(os.path.join(out_dir, fname), seq)
        rates = detection_rates(seq)
        low = is_low_quality(rates)
        last_saved = {"fname": fname, "rates": rates, "low_quality": low}
        print(
            f"  saved {fname} {seq.shape} | pose {rates['pose']:.0%} | left {rates['lh']:.0%} | "
            f"right {rates['rh']:.0%} | face {rates['face']:.0%}"
        )
        if low:
            print(
                "  low detection: keep head, torso and hands in frame and check the lighting; "
                "consider re-recording this take."
            )
        write_status(
            status_file, state="saved", sample_idx=sample_idx + 1, last_saved=last_saved, **progress
        )

        if sample_idx < n_samples - 1:
            for _ in range(REST_FRAMES):
                if should_stop(stop_file):
                    return
                ret, frame = cap.read()
                if not ret:
                    break
                _draw_hud(
                    cv2,
                    frame,
                    word=word,
                    sample_idx=sample_idx,
                    n_samples=n_samples,
                    mode="rest",
                    last_saved=last_saved,
                )
                cv2.imshow(WINDOW, frame)
                cv2.waitKey(REST_FRAME_MS)
            write_status(
                status_file,
                state="rest",
                sample_idx=sample_idx + 1,
                last_saved=last_saved,
                **progress,
            )


def _check_outside_repo(path: str) -> None:
    out = os.path.abspath(path)
    if os.path.commonpath([out, _REPO_ROOT]) == _REPO_ROOT:
        raise SystemExit(
            f"--out {out} is inside the repository. Recorded keypoints are personal data; "
            "store them outside the repository."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="Root folder for raw takes (outside the repo)")
    ap.add_argument("--signer-id", required=True, help="Anonymous signer id, e.g. signer_00")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--class", dest="single_class", help="A single class to record, e.g. AGUA")
    g.add_argument("--classes", help="Comma-separated classes, e.g. HOLA,GRACIAS,AGUA")
    ap.add_argument("--n-samples", type=int, default=15, help="Takes per class")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument(
        "--model-complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe pose model (2 = better at distance, slower)",
    )
    ap.add_argument("--status-file", default=None, help="JSON file for live progress (demo app)")
    ap.add_argument("--stop-file", default=None, help="Stop after the current take if this exists")
    args = ap.parse_args()
    _check_outside_repo(args.out)

    raw = args.classes.split(",") if args.classes else [args.single_class]
    classes = [c.strip().upper() for c in raw]
    out_dir = os.path.join(args.out, args.signer_id)

    import cv2
    import mediapipe as mp

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    if not cap.isOpened():
        write_status(args.status_file, state="error", error=f"Could not open camera {args.camera}")
        raise SystemExit(f"Could not open camera {args.camera}")

    write_status(args.status_file, state="starting", classes_pending=classes, classes_done=[])
    done: list[str] = []
    try:
        with mp_holistic.Holistic(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            refine_face_landmarks=True,
            model_complexity=args.model_complexity,
        ) as holistic:
            for idx, word in enumerate(classes):
                if should_stop(args.stop_file):
                    break
                record_class(
                    holistic,
                    cap,
                    cv2,
                    mp_holistic,
                    mp_drawing,
                    word,
                    args.n_samples,
                    out_dir,
                    status_file=args.status_file,
                    stop_file=args.stop_file,
                    classes_done=done,
                    classes_pending=classes[idx + 1 :],
                )
                done.append(word)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    stopped = should_stop(args.stop_file)
    write_status(
        args.status_file,
        state="stopped" if stopped else "done",
        classes_done=done,
        classes_pending=[],
    )
    print(f"\nDone. Takes in {out_dir}/ ({C.N}-node layout, {C.RAW_FEATURES} values per frame).")


if __name__ == "__main__":
    main()
