"""Dataset explorer: take quality plus frame-by-frame replay of the recorded skeleton."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from lib import inventory, paths, viz

st.set_page_config(page_title="Explore | LSP", page_icon=":material/folder_open:", layout="wide")
st.title("Explore the recorded dataset")

raw_dir = paths.require_raw_dir()
if raw_dir is None:
    st.warning(
        "Set `LSP_RAW_DIR` to a folder outside the repository first.", icon=":material/warning:"
    )
    st.stop()


@st.cache_data(show_spinner=False)
def _quality(sample: inventory.Sample) -> dict:
    return inventory.sample_quality(sample)


signers = inventory.list_signers(str(raw_dir))
if not signers:
    st.info("No takes recorded yet. Go to **Record** to start a session.")
    st.stop()

with st.sidebar:
    signer_id = st.selectbox("Signer", signers)
    samples = inventory.list_samples(str(raw_dir), signer_id)
    labels = sorted({s.label for s in samples})
    label_filter = st.selectbox("Class", ["(all)", *labels])
    only_low = st.checkbox("Show only low-detection takes", value=False)

filtered = [s for s in samples if label_filter == "(all)" or s.label == label_filter]
st.subheader(f"{signer_id}: {len(samples)} takes, {len(labels)} classes")
st.bar_chart(pd.Series(inventory.class_counts(str(raw_dir), signer_id), name="takes"))

rows = []
for s in filtered:
    q = _quality(s)
    rows.append(
        {
            "class": s.label,
            "file": s.fname,
            "shape": str(q["shape"]),
            "pose": q["rates"]["pose"],
            "left_hand": q["rates"]["lh"],
            "right_hand": q["rates"]["rh"],
            "face": q["rates"]["face"],
            "low_quality": q["low_quality"],
        }
    )
if only_low:
    rows = [r for r in rows if r["low_quality"]]
pct = {"min_value": 0, "max_value": 1, "format": "%.0f%%"}
st.dataframe(
    rows,
    width="stretch",
    hide_index=True,
    column_config={
        "pose": st.column_config.ProgressColumn("pose", **pct),
        "left_hand": st.column_config.ProgressColumn("left hand", **pct),
        "right_hand": st.column_config.ProgressColumn("right hand", **pct),
        "face": st.column_config.ProgressColumn("face", **pct),
        "low_quality": st.column_config.CheckboxColumn("low quality"),
    },
)

st.divider()
st.subheader("Replay a take")
if not filtered:
    st.caption("No take matches the current filter.")
    st.stop()

chosen = st.selectbox("Take", [s.fname for s in filtered])
sample = next(s for s in filtered if s.fname == chosen)
seq = inventory.load_sample(sample)
q = _quality(sample)

left, right = st.columns([2, 1])
with left:
    if not q["ok_shape"]:
        st.warning(f"This take has shape `{q['shape']}`, not the 73-node raw layout `(30, 285)`.")
    else:
        st.session_state.setdefault("frame_idx", 0)
        st.session_state.setdefault("playing", False)
        # Every change to frame_idx must happen before the slider with the same key is created:
        # Streamlit forbids writing a widget's state after it has been instantiated in the same run.
        st.session_state.frame_idx = min(st.session_state.frame_idx, seq.shape[0] - 1)
        if st.session_state.playing:
            st.session_state.frame_idx = (st.session_state.frame_idx + 1) % seq.shape[0]
        pcol, scol = st.columns([1, 5])
        with pcol:
            playing = st.session_state.playing
            if st.button(
                "Pause" if playing else "Play",
                icon=":material/pause:" if playing else ":material/play_arrow:",
            ):
                st.session_state.playing = not playing
        with scol:
            st.slider("Frame", 0, seq.shape[0] - 1, key="frame_idx")
        st.pyplot(viz.skeleton_figure(seq, st.session_state.frame_idx))
        if st.session_state.playing:
            time.sleep(0.12)
            st.rerun()

with right:
    st.markdown(f"**{sample.label}**: `{sample.fname}`")
    if q["ok_shape"]:
        st.pyplot(viz.detection_bar_figure(q["rates"]))
    if q["low_quality"]:
        st.warning("Low detection in this take.", icon=":material/warning:")
    confirm = st.checkbox("Confirm before moving to the trash")
    if st.button("Move to trash", icon=":material/delete:", disabled=not confirm):
        dest = inventory.move_to_trash(sample, str(raw_dir))
        st.cache_data.clear()
        st.success(f"Moved to `{dest}`", icon=":material/check_circle:")
        time.sleep(0.6)
        st.rerun()
    st.caption("Nothing is deleted: the take goes to `_trash/` inside the signer's folder.")
