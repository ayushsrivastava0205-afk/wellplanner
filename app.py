import streamlit as st
import sys
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Import the core well planning module
try:
    from well_planner_core import run_well_plan, plot_3d
except ImportError:
    st.error("❌ Error: Could not import well_planner_core. Make sure well_planner_core.py is in the repo.")
    st.stop()

st.set_page_config(page_title="Well Planner", layout="wide")

st.title("🛢️ Directional Well Planner")
st.markdown("""
Interactive well trajectory design tool with:
- Coordinate conversion (lat/lon ↔ northing/easting)
- Minimum curvature trajectory computation
- 3D visualization and planning views
- Interactive folium maps
- Survey uncertainty analysis
""")

st.sidebar.header("⚙️ Well Configuration")

# Sidebar inputs
well_name = st.sidebar.text_input("Well Name", value="Well-1")
units = st.sidebar.selectbox("Units", ["metric", "imperial"], index=0)
crs = st.sidebar.text_input("CRS (EPSG)", value="EPSG:32630")

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Surface Location")
surface_lat = st.sidebar.number_input("Surface Latitude (°)", value=55.0, format="%.6f")
surface_lon = st.sidebar.number_input("Surface Longitude (°)", value=2.5, format="%.6f")

st.sidebar.subheader("🎯 Target Location")
target_lat = st.sidebar.number_input("Target Latitude (°)", value=55.05, format="%.6f")
target_lon = st.sidebar.number_input("Target Longitude (°)", value=2.6, format="%.6f")
target_tvd = st.sidebar.number_input("Target TVD (m)", value=2000.0)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Trajectory Design")
kop_tvd = st.sidebar.number_input("KOP Depth (m)", value=500.0)
build_rate = st.sidebar.number_input("Build Rate (°/100m)", value=2.0, step=0.1)
surface_dls = st.sidebar.number_input("Surface DLS (°/100m)", value=0.0, step=0.1)
survey_interval = st.sidebar.number_input("Survey Interval (m)", value=30.0)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Survey Tool")
survey_tool = st.sidebar.selectbox(
    "Survey Tool",
    ["MWD", "MWD_IFR", "GYRO_SS", "GYRO_CONT", "SINGLE_SHOT"],
    index=0
)
confidence = st.sidebar.slider("Confidence Level (σ)", 1.0, 3.0, 2.0, step=0.5)

# Main area
col1, col2 = st.columns([3, 1])

with col2:
    calculate_btn = st.button("🚀 Calculate Trajectory", type="primary", use_container_width=True)

if calculate_btn:
    with st.spinner("Computing trajectory..."):
        try:
            # Build config dictionary for run_well_plan()
            config = {
                'well': {
                    'name': well_name
                },
                'location': {
                    'crs': crs,
                    'surface': {
                        'lat': surface_lat,
                        'lon': surface_lon
                    },
                    'target': {
                        'lat': target_lat,
                        'lon': target_lon,
                        'tvd': target_tvd
                    }
                },
                'trajectory': {
                    'units': units,
                    'kop_tvd': kop_tvd,
                    'build_rate': build_rate,
                    'surface_dls': surface_dls,
                    'survey_interval': survey_interval
                },
                'survey': {
                    'tool_sections': [
                        {'from_md': 0, 'to_md': None, 'code': survey_tool}
                    ],
                    'confidence': confidence
                }
            }

            # Run the well planning calculation
            # run_well_plan returns (survey_df, traj, uncertainty_df, fig, map_obj)
            survey_df, traj, uncertainty_df, fig, map_obj = run_well_plan(config)

            st.success("✅ Trajectory computed successfully!")

            # Display results in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Survey Data", "📈 Plots", "🗺️ Map", "📥 Export"])

            with tab1:
                st.subheader("Survey Station Table")
                st.dataframe(survey_df.head(20), use_container_width=True)
                st.caption(f"Showing first 20 of {len(survey_df)} stations")

                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total MD", f"{survey_df['MD'].max():.1f} m")
                with col2:
                    st.metric("Max Inclination", f"{survey_df['Inc'].max():.2f}°")
                with col3:
                    st.metric("Target TVD", f"{target_tvd:.1f} m")
                with col4:
                    st.metric("Stations", len(survey_df))

            with tab2:
                st.subheader("Well Trajectory Visualizations")
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info("📊 3D plot generation requires matplotlib")

            with tab3:
                st.subheader("Interactive Map")
                if map_obj is not None:
                    st.write("Map object ready for integration")
                else:
                    st.info("🗺️ Map requires folium setup")

            with tab4:
                st.subheader("Export Data")
                
                # CSV export
                csv_buffer = BytesIO()
                survey_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Survey Data (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name=f"{well_name}_survey.csv",
                    mime="text/csv"
                )

                st.info("✅ Survey data ready for download")

        except Exception as e:
            st.error(f"❌ Error during calculation: {str(e)}")
            st.write("Debug info:", e)

st.markdown("---")
st.markdown("""
### About this tool
Built with Streamlit + well_planner_core — a professional directional well design system.
- Minimum curvature survey computation
- ISCWSA-compatible uncertainty analysis
- Multi-zone trajectory design
""")
