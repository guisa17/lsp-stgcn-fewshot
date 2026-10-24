"""Streamlit demo: record LSP takes, explore them, track progress and test trained models.

Run from the repository root:
    pip install -e ".[demo]"
    LSP_RAW_DIR=/path/outside/repo/raw streamlit run demo/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from lib import inventory, paths, vocab

st.set_page_config(page_title="LSP demo", page_icon=":material/sign_language:", layout="wide")
st.title("LSP recorder and model demo")
st.caption(
    "Peruvian Sign Language recognition with ST-GCN / HA-GCN. Record webcam sessions as keypoints, "
    "check take quality, track dataset progress and test trained models."
)

raw_dir = paths.require_raw_dir()
if raw_dir is None:
    st.warning(
        "Set the environment variable `LSP_RAW_DIR` to a folder **outside the repository** before "
        "starting the app. Recorded keypoints include facial landmarks and are personal data.",
        icon=":material/warning:",
    )
    st.stop()

voc = vocab.load_vocab()
inv = inventory.full_inventory(str(raw_dir))
signers = sorted(inv)
total = sum(sum(c.values()) for c in inv.values())
n_seen = len({label for counts in inv.values() for label in counts})
target_total = voc["target_takes_per_signer"] * len(voc["classes"]) * max(len(signers), 1)

for col, value, label in zip(
    st.columns(4),
    [
        str(len(signers)),
        str(total),
        f"{n_seen}/{len(voc['classes'])}",
        f"{min(total / target_total, 1.0):.0%}" if target_total else "-",
    ],
    [
        "signers with data",
        "recorded takes",
        "vocabulary classes with at least one take",
        "progress towards the target",
    ],
    strict=False,
):
    col.metric(label, value)

st.subheader("Pages")
for col, (title, text, page, icon) in zip(
    st.columns(4),
    [
        (
            "Record",
            "Choose signer, classes and takes; record with an on-screen countdown and live "
            "detection quality.",
            "pages/1_Record.py",
            ":material/videocam:",
        ),
        (
            "Explore dataset",
            "Replay recorded skeletons frame by frame, filter by quality and move bad "
            "takes to a trash folder.",
            "pages/2_Explore_Dataset.py",
            ":material/folder_open:",
        ),
        (
            "Progress",
            "Signer x class matrix against the target number of takes.",
            "pages/3_Progress.py",
            ":material/monitoring:",
        ),
        (
            "Test model",
            "Run a trained model on a recorded take or live from the camera.",
            "pages/4_Test_Model.py",
            ":material/model_training:",
        ),
    ],
    strict=False,
):
    with col:
        st.markdown(f"**{title}**")
        st.caption(text)
        st.page_link(page, label=f"Open {title}", icon=icon)

st.divider()
with st.expander("Environment (camera / MediaPipe)"):
    for name in ("cv2", "mediapipe"):
        try:
            __import__(name)
            st.success(f"**{name}** available", icon=":material/check_circle:")
        except ImportError as e:
            st.error(
                f'**{name}**: `{e}`. Install with `pip install -e ".[demo]"`.',
                icon=":material/cancel:",
            )
    st.caption(f"Interpreter used for recording: `{paths.python_executable()}`")
st.caption(
    "The video window opens outside the browser (a native OpenCV window): the app captures the "
    "webcam of the machine running the Streamlit server."
)
