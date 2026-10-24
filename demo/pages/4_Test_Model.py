"""Test page: run a trained model on a recorded take or live from the camera."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from lib import inventory, paths, sessions, viz

from lsp import inference

st.set_page_config(
    page_title="Test model | LSP", page_icon=":material/model_training:", layout="wide"
)
st.title("Test a model")
st.caption(
    f"Runs trained with `scripts/train.py`, read from `{paths.RUNS_DIR}` (set `LSP_RUNS_DIR`)."
)


@st.cache_resource(show_spinner="Loading model...")
def _load(name: str) -> inference.LoadedRun:
    return inference.load_run(str(paths.RUNS_DIR), name)


run_names = inference.list_runs(str(paths.RUNS_DIR))
if not run_names:
    st.info(
        "No trained runs found. Train one first: "
        "`python scripts/train.py --config configs/lsp/hagcn.yaml`."
    )
    st.stop()

top_l, top_r = st.columns([4, 1])
with top_l:
    run_name = st.selectbox("Run", run_names)
with top_r:
    st.write("")
    if st.button("Reload", icon=":material/refresh:"):
        st.cache_resource.clear()
        st.rerun()

try:
    run = _load(run_name)
except Exception as e:
    st.error(str(e), icon=":material/error:")
    st.stop()

m = run.metrics
if m:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("test accuracy", f"{m.get('test_metrics', {}).get('accuracy', 0):.1%}")
    c2.metric("test F1", f"{m.get('test_metrics', {}).get('f1', 0):.2f}")
    c3.metric("CPU latency", f"{m.get('latency', {}).get('latency_ms', 0):.0f} ms")
    c4.metric("classes", len(run.classes))
st.caption(f"streams: {', '.join(run.streams)} | architecture: {run.cfg['model']['type']}")

st.divider()
tab_sample, tab_live = st.tabs(["Recorded take", "Live camera"])

with tab_sample:
    raw_dir = paths.require_raw_dir()
    signers = inventory.list_signers(str(raw_dir)) if raw_dir else []
    if not signers:
        st.info("No recorded takes found (set `LSP_RAW_DIR`, or record some with **Record**).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            signer_id = st.selectbox("Signer", signers, key="test_signer")
        samples = [
            s
            for s in inventory.list_samples(str(raw_dir), signer_id)
            if inventory.sample_quality(s)["ok_shape"]
        ]
        if not samples:
            st.warning("This signer has no takes in the 73-node layout `(30, 285)`.")
        else:
            with c2:
                chosen = st.selectbox("Take", [s.fname for s in samples], key="test_sample")
            sample = next(s for s in samples if s.fname == chosen)
            seq = inventory.load_sample(sample)
            if st.button("Classify this take", icon=":material/play_arrow:", type="primary"):
                result = run.predict_raw(seq)
                correct = result["pred"] == sample.label
                left, right = st.columns([1, 2])
                with left:
                    st.pyplot(viz.skeleton_figure(seq, seq.shape[0] // 2))
                with right:
                    (st.success if correct else st.error)(
                        f"true: **{sample.label}**, predicted: **{result['pred']}** "
                        f"({result['confidence']:.0%})",
                        icon=":material/check_circle:" if correct else ":material/cancel:",
                    )
                    st.pyplot(viz.topk_bar_figure(result["ranked"], true_label=sample.label))

with tab_live:
    st.info(
        "The camera opens in a **separate window** (local webcam of the server). This tab "
        "mirrors the live prediction.",
        icon=":material/info:",
    )
    session = st.session_state.get("live_session")
    if session is not None and session.poll() is None:

        @st.fragment(run_every=0.5)
        def _render_live() -> None:
            sess = st.session_state.get("live_session")
            if sess is None:
                return
            status = sess.read_status()
            state = status.get("state", "loading")
            if state == "predicting" and status.get("ranked"):
                st.subheader(
                    f"prediction: {status.get('pred', '-')} ({status.get('confidence', 0):.0%})"
                )
                st.pyplot(viz.topk_bar_figure(status["ranked"]))
            elif state == "error":
                st.error(status.get("error", "unknown error"), icon=":material/error:")
            else:
                st.caption(
                    {
                        "loading": "loading model and camera...",
                        "ready": "waiting for a 30-frame buffer...",
                    }.get(state, state)
                )
            if st.button("Stop", icon=":material/stop_circle:"):
                sess.request_stop()
                st.toast("Stopping...")
            if sess.poll() is not None:
                sess.cleanup()
                del st.session_state["live_session"]
                st.rerun()

        _render_live()
    else:
        st.session_state.pop("live_session", None)
        c1, c2, c3 = st.columns(3)
        with c1:
            camera = st.number_input(
                "Camera index", min_value=0, max_value=5, value=0, step=1, key="live_camera"
            )
        with c2:
            complexity = st.select_slider(
                "Detection quality", options=[0, 1, 2], value=1, key="live_complexity"
            )
        with c3:
            interval = st.slider("Seconds between predictions", 0.3, 1.5, 0.6, 0.1)
        if st.button("Start live classification", icon=":material/play_arrow:", type="primary"):
            st.session_state["live_session"] = sessions.start_live_classifier(
                run=run_name,
                camera=int(camera),
                model_complexity=int(complexity),
                interval=float(interval),
            )
            st.rerun()
