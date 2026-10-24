"""Progress page: signer x class matrix against the vocabulary target."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st
from lib import inventory, paths, vocab

st.set_page_config(page_title="Progress | LSP", page_icon=":material/monitoring:", layout="wide")
st.title("Dataset progress")

raw_dir = paths.require_raw_dir()
if raw_dir is None:
    st.warning(
        "Set `LSP_RAW_DIR` to a folder outside the repository first.", icon=":material/warning:"
    )
    st.stop()

voc = vocab.load_vocab()
target = voc["target_takes_per_signer"]
vocab_classes = voc["classes"]
inv = inventory.full_inventory(str(raw_dir))
signers = sorted(inv)
if not signers:
    st.info("No takes recorded yet. Go to **Record** to start a session.")
    st.stop()

extra_classes = sorted({c for counts in inv.values() for c in counts} - set(vocab_classes))
all_classes = vocab_classes + extra_classes
matrix = pd.DataFrame(
    {s: [inv[s].get(c, 0) for c in all_classes] for s in signers}, index=all_classes
).T  # rows = signers, columns = classes

st.caption(f"Target in `demo/vocab.yaml`: **{target} takes** per class and signer.")


def _highlight(v: int) -> str:
    if v == 0:
        return "background-color: rgba(220,60,60,0.35)"
    if v < target:
        return "background-color: rgba(230,180,40,0.35)"
    return "background-color: rgba(70,170,90,0.35)"


st.dataframe(matrix.style.map(_highlight), width="stretch")
st.download_button(
    "Download matrix (CSV)",
    icon=":material/download:",
    data=matrix.to_csv().encode("utf-8"),
    file_name="dataset_progress.csv",
    mime="text/csv",
)

st.divider()
st.subheader("Progress per signer")
for signer in signers:
    counts = inv[signer]
    complete = sum(1 for c in vocab_classes if counts.get(c, 0) >= target)
    takes = sum(counts.get(c, 0) for c in vocab_classes)
    target_total = target * len(vocab_classes)
    pending = [c for c in vocab_classes if counts.get(c, 0) < target]
    with st.expander(
        f"**{signer}**: {complete}/{len(vocab_classes)} classes complete "
        f"({takes}/{target_total} takes)",
        icon=":material/check_circle:" if not pending else ":material/pending:",
    ):
        st.progress(min(takes / target_total, 1.0) if target_total else 0.0)
        if pending:
            st.dataframe(
                pd.DataFrame(
                    {
                        "class": pending,
                        "takes": [counts.get(c, 0) for c in pending],
                        "missing": [max(target - counts.get(c, 0), 0) for c in pending],
                    }
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.success("Vocabulary complete for this signer.", icon=":material/check_circle:")

if extra_classes:
    st.divider()
    st.subheader("Classes recorded outside the vocabulary")
    st.caption("Add them to `demo/vocab.yaml` if they belong to the final vocabulary.")
    st.write(", ".join(extra_classes))
