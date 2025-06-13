import pandas as pd
import streamlit as st

from src.app.constants import RECOMMEND_POST_BUTTON, TOXICITY_STRICTNESS
from src.data.constants import DataCols
from src.engine.run_engine import FeedShiftEngine
from src.utils.tools import load_csv


# ---------------- Page Setup ----------------
st.set_page_config(page_title="FeedShift: Rewiring your feed", layout="wide")
st.markdown(
    """
    <style>
    .centered-box {
        margin: auto;
        width: 50%;
        padding: 5px;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #f9f9f9;
        box-shadow: 0px 0px 12px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .rounded-container {
        margin: auto;
        width: 60%;
        padding: 2px;
        border-radius: 12px;
        background-color: #fdfdfd;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
        border: 5px solid #e2e2e2;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>📢 FeedShift: Rewiring your feed</h1>",
    unsafe_allow_html=True,
)

# ---------------- Centered Upload Box ----------------
with st.container():
    st.markdown("<div class='centered-box'>", unsafe_allow_html=True)
    st.subheader("📂 Upload Feed CSV")
    uploaded_file = st.file_uploader("", type=["csv"], key="upload_box")
    st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file:
    if (
        "uploaded_file_name" in st.session_state
        and uploaded_file.name != st.session_state["uploaded_file_name"]
    ):
        st.session_state[RECOMMEND_POST_BUTTON] = False
        st.session_state["uploaded_file_name"] = uploaded_file.name
    elif "uploaded_file_name" not in st.session_state:
        st.session_state["uploaded_file_name"] = uploaded_file.name

# ---------------- Slider ----------------
if uploaded_file:
    with st.container():
        tox_strict = st.slider(
            "🛡️ Toxicity Filter Strictness",  # Add proper label
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher values demote more toxic content",
            key=TOXICITY_STRICTNESS,
        )
        st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource
def get_recommended_posts(
    data: pd.DataFrame, toxicity_strictness: float
) -> pd.DataFrame:
    with st.spinner("🎩 Abra-Ka-Dabra ..."):
        engine = FeedShiftEngine(data)
        ranked_data = engine.run(toxicity_strictness).head(10)
    return ranked_data


def _recommend_post_button():
    st.session_state[RECOMMEND_POST_BUTTON] = True


def run():
    if uploaded_file:
        data = load_csv(uploaded_file)

        if not st.session_state.get(RECOMMEND_POST_BUTTON, False):
            if st.button("🚀 Recommend Top 10 Posts", on_click=_recommend_post_button):
                st.rerun()

        if st.session_state.get(RECOMMEND_POST_BUTTON, False):
            ranked_data = get_recommended_posts(
                data, st.session_state[TOXICITY_STRICTNESS]
            )
            st.markdown("---")
            st.subheader("🔝 Top 10 Recommended Posts", divider="rainbow")
            st.markdown("")
            st.markdown("")
            st.markdown("")

            for _, row in ranked_data.iterrows():
                with st.container():
                    col_content, col_likes, col_score = st.columns([8, 1, 1])

                    with col_content:
                        st.markdown(
                            f"👤 **{row[DataCols.AUTHOR]}**", unsafe_allow_html=True
                        )
                        st.markdown(
                            f"🕒 {row[DataCols.TIMESTAMP][:10]} • 🌐 {row[DataCols.PLATFORM].capitalize()}",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='border:1px solid #ddd; border-radius:8px; padding:12px; margin-top:8px;'>"
                            f"{row[DataCols.TEXT]}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"🏷️ {row[DataCols.POST_ID]}", unsafe_allow_html=True
                        )

                    with col_likes:
                        st.markdown(
                            f"❤️\n\n{row[DataCols.LIKES]}", unsafe_allow_html=True
                        )

                    with col_score:
                        st.markdown(
                            f"⭐\n\n{row[DataCols.SCORES]}", unsafe_allow_html=True
                        )
                    st.markdown(
                        "<div class='rounded-container'>", unsafe_allow_html=True
                    )
                    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    run()
