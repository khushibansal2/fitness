import importlib.util
import json
import os
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_PATH = PROJECT_ROOT / "fitness_analyzer (2).py"


@st.cache_resource
def load_analyzer():
    spec = importlib.util.spec_from_file_location("fitness_analyzer_module", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the fitness analyzer module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnhancedFitnessAnalyzer()


st.set_page_config(page_title="Fitness Analyzer", page_icon="🏃", layout="wide")
st.title("Fitness Analyzer")
st.write("Upload a video and get an AI-powered fitness assessment for sit-ups, vertical jump, broad jump, or flexibility.")

with st.sidebar:
    st.header("Analysis settings")
    test_type = st.selectbox(
        "Test type",
        ["situps", "vertical_jump", "broad_jump", "flexibility"],
        index=0,
    )
    age_group = st.selectbox("Age group", ["teenage", "youth", "adult"], index=1)
    gender = st.selectbox("Gender", ["male", "female", "other"], index=0)
    show_overlay = st.checkbox("Show pose overlay", value=True)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Run analysis", type="primary"):
        with st.spinner("Analyzing video... This may take a few minutes."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix or ".mp4") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                analyzer = load_analyzer()

                if test_type == "situps":
                    result = analyzer.analyze_situps(temp_path, age_group, gender, show_overlay)
                elif test_type == "vertical_jump":
                    result = analyzer.analyze_vertical_jump(temp_path, age_group, gender, show_overlay)
                elif test_type == "broad_jump":
                    result = analyzer.analyze_broad_jump(temp_path, age_group, gender, show_overlay)
                else:
                    result = analyzer.analyze_flexibility(temp_path, age_group, gender, show_overlay)

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Analysis complete")
                    st.subheader("Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test type", result["test_type"].replace("_", " ").title())
                        if "raw_count" in result:
                            st.metric("Repetitions", result["raw_count"])
                        elif "raw_height_cm" in result:
                            st.metric("Jump height", f"{result['raw_height_cm']} cm")
                        elif "raw_distance_cm" in result:
                            st.metric("Jump distance", f"{result['raw_distance_cm']} cm")
                        else:
                            st.metric("Reach", f"{result['raw_reach_cm']} cm")
                    with col2:
                        st.metric("Score", f"{result['score']}/100")
                        st.metric("Rating", result["feedback"]["overall_rating"])

                    st.write("### Technique tips")
                    st.write(result["feedback"]["technique_tips"])
                    st.write("### Improvement target")
                    st.write(result["feedback"]["improvement_targets"])

                    st.download_button(
                        label="Download JSON results",
                        data=json.dumps(result, indent=2),
                        file_name=f"{test_type}_results.json",
                        mime="application/json",
                    )
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Analysis failed: {exc}")
else:
    st.info("Choose a video file to begin.")
