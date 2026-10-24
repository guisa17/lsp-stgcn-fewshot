"""Record page: configure a session and launch the recorder (native video window)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from lib import inventory, paths, sessions, viz, vocab

st.set_page_config(page_title="Record | LSP", page_icon=":material/videocam:", layout="wide")
st.title("Record a session")

raw_dir = paths.require_raw_dir()
if raw_dir is None:
    st.warning(
        "Set `LSP_RAW_DIR` to a folder outside the repository first.", icon=":material/warning:"
    )
    st.stop()

voc = vocab.load_vocab()
known_signers = inventory.list_signers(str(raw_dir))

STATE_LABELS = {
    "starting": "Starting camera and MediaPipe...",
    "countdown": "Countdown",
    "recording": "Recording",
    "saved": "Take saved",
    "discarded": "Take discarded (unexpected shape)",
    "rest": "Rest",
    "stopped": "Stopped by the user",
    "done": "Session complete",
    "error": "Error",
}


def _render_form() -> None:
    st.info(
        "The camera opens in a **separate window**, not in the browser: this app runs on the same "
        "machine as the webcam. Record only with the signer's informed consent."
    )
    with st.expander("Test the camera before recording"):
        cam_test = st.number_input(
            "Camera index to test", min_value=0, max_value=5, value=0, step=1
        )
        if st.button("Take a test snapshot", icon=":material/photo_camera:"):
            import cv2

            cap = cv2.VideoCapture(int(cam_test))
            ok, frame = cap.read()
            cap.release()
            if ok:
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Camera {cam_test}", width=320
                )
                st.caption("The snapshot is only shown here; it is not saved.")
            else:
                st.error(
                    f"Could not read camera {cam_test}. Try another index.", icon=":material/error:"
                )

    st.subheader("Session settings")
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio(
            "Signer",
            ["Existing", "New"],
            horizontal=True,
            disabled=not known_signers,
            index=0 if known_signers else 1,
        )
        if mode == "Existing" and known_signers:
            signer_id = st.selectbox("Signer", known_signers)
        else:
            signer_id = st.text_input(
                "Anonymous id of the new signer (e.g. signer_02)", value="signer_02"
            )
        camera = st.number_input(
            "Camera index", min_value=0, max_value=5, value=int(cam_test), step=1
        )
        model_complexity = st.select_slider(
            "MediaPipe detection quality",
            options=[0, 1, 2],
            value=1,
            help="2 = better detection at distance but slower; 0 = faster, less accurate.",
        )
    with c2:
        vocab_classes = st.multiselect("Vocabulary classes to record", options=voc["classes"])
        extra_raw = st.text_input(
            "Extra classes (comma-separated, optional)", placeholder="e.g. NEW_SIGN, OTHER_SIGN"
        )
        n_samples = st.number_input(
            "Takes per class", min_value=1, max_value=50, value=voc["target_takes_per_signer"]
        )

    extra = [c.strip().upper() for c in extra_raw.split(",") if c.strip()]
    classes = list(dict.fromkeys(vocab_classes + extra))  # de-duplicate, keep order
    if signer_id and classes:
        current = inventory.class_counts(str(raw_dir), signer_id)
        st.caption(
            f"Already recorded for **{signer_id}**: "
            + ", ".join(f"{c} ({current.get(c, 0)})" for c in classes)
        )

    disabled = not signer_id or not classes
    if st.button(
        "Start recording",
        icon=":material/play_arrow:",
        type="primary",
        disabled=disabled,
        width="stretch",
    ):
        st.session_state["rec_session"] = sessions.start_recording(
            raw_dir=str(raw_dir),
            signer_id=signer_id,
            classes=classes,
            n_samples=int(n_samples),
            camera=int(camera),
            model_complexity=int(model_complexity),
        )
        st.session_state["rec_signer"] = signer_id
        st.rerun()
    if disabled:
        st.caption("Choose a signer and at least one class to enable the button.")


@st.fragment(run_every=1.0)
def _render_live() -> None:
    session = st.session_state.get("rec_session")
    if session is None:
        return
    status = session.read_status()
    finished = session.poll() is not None
    state = status.get("state", "starting")
    done = status.get("classes_done") or []
    pending = status.get("classes_pending") or []

    st.subheader(f"Recording **{st.session_state.get('rec_signer', '?')}**")
    st.progress(
        min(len(done) / max(len(done) + len(pending) + 1, 1), 1.0),
        text=STATE_LABELS.get(state, state),
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Current class", status.get("clase", "-"))
    c2.metric("Take", f"{status.get('sample_idx', 0)}/{status.get('n_samples', '-')}")
    if state == "recording":
        c3.metric("Frame", f"{status.get('frame_num', 0)}/{status.get('seq_length', 30)}")
    elif state == "countdown":
        c3.metric("Countdown", status.get("countdown", "-"))
    else:
        c3.metric("State", STATE_LABELS.get(state, state))

    last = status.get("last_saved")
    if last:
        low = last.get("low_quality")
        (st.warning if low else st.success)(
            f"Last saved take: `{last['fname']}`"
            + (" (low detection, consider re-recording)" if low else ""),
            icon=":material/warning:" if low else ":material/check_circle:",
        )
        st.pyplot(viz.detection_bar_figure(last["rates"]))
    if done or pending:
        st.caption("Done: " + (", ".join(done) or "-"))
        st.caption("Pending: " + (", ".join(pending) or "-"))

    if st.button("Stop session", icon=":material/stop_circle:"):
        session.request_stop()
        st.toast("Stopping after the current take...", icon=":material/info:")

    if finished:
        if status.get("state") == "stopped":
            st.info("Session stopped by the user.", icon=":material/info:")
        elif status.get("error"):
            st.error(f"Error: {status['error']}", icon=":material/error:")
        else:
            st.success("Session complete.", icon=":material/check_circle:")
        session.cleanup()
        del st.session_state["rec_session"]
        time.sleep(1.0)
        st.rerun()


if "rec_session" in st.session_state and st.session_state["rec_session"].poll() is None:
    _render_live()
    st.divider()
    st.caption("You can leave this page; recording continues in its own window.")
else:
    st.session_state.pop("rec_session", None)
    _render_form()
