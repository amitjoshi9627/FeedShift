import pandas as pd
import streamlit as st

from src.data.constants import DataCols
from src.engine.run_engine import FeedShiftEngine
from src.utils.tools import load_csv

st.set_page_config(page_title="FeedShift: Rewiring your feed", layout="wide")
st.title("📢 FeedShift: Rewiring your feed")

uploaded_file = st.file_uploader("📂 Upload Feed CSV", type=["csv"])

if uploaded_file and st.button("🚀 Recommend Top 10 Posts"):
    with st.spinner("🎩 Abra-Ka-Dabra ..."):
        engine = FeedShiftEngine(load_csv(uploaded_file))
        ranked_data = engine.run().head(10)

    st.markdown("---")
    st.subheader("🔝 Top 10 Recommended Posts")

    for _, row in ranked_data.iterrows():
        card = st.container()
        col_content, col_likes, col_score = card.columns([8, 1, 1])

        # Main content + metadata
        with col_content:
            # Author line with emoji
            st.markdown(f"👤 **{row[DataCols.AUTHOR]}**", unsafe_allow_html=True)
            # Timestamp + platform
            st.markdown(
                f"🕒 {row[DataCols.TIMESTAMP][:10]} • 🌐 {row[DataCols.PLATFORM].capitalize()}",
                unsafe_allow_html=True
            )
            # Bordered text box
            st.markdown(
                f"<div style='border:1px solid #ddd; border-radius:8px; padding:12px; margin-top:8px;'>"
                f"{row[DataCols.TEXT]}"
                f"</div>",
                unsafe_allow_html=True
            )
            # Post ID with tag emoji
            st.markdown(
                f"🏷️ {row[DataCols.POST_ID]}",
                unsafe_allow_html=True
            )

        # Likes column
        with col_likes:
            st.markdown(f"❤️\n\n{row[DataCols.LIKES]}", unsafe_allow_html=True)

        # Score column
        with col_score:
            st.markdown(f"⭐\n\n{row[DataCols.SCORES]}", unsafe_allow_html=True)

        st.markdown("---")
