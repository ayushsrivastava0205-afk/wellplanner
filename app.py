import streamlit as st
import sys
import os

st.set_page_config(page_title="Well Planner", layout="wide")

st.title("🛢️ Directional Well Planner")
st.markdown("""
Welcome to the interactive Well Planning tool.
This application helps you design and visualize well trajectories with:
- Coordinate conversion (lat/lon ↔ northing/easting)
- Minimum curvature trajectory computation
- Interactive 3D visualization
- Plan and vertical section views
- Export to folium maps, CSV, and Excel
""")

st.header("Well Parameters")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Surface Location")
    surface_lat = st.number_input("Surface Latitude (°)", value=0.0, format="%.6f")
    surface_lon = st.number_input("Surface Longitude (°)", value=0.0, format="%.6f")

with col2:
    st.subheader("Target Location")
    target_lat = st.number_input("Target Latitude (°)", value=0.01, format="%.6f")
    target_lon = st.number_input("Target Longitude (°)", value=0.01, format="%.6f")

col1, col2, col3 = st.columns(3)

with col1:
    surface_tvd = st.number_input("Surface TVD (m)", value=0.0)
    target_tvd = st.number_input("Target TVD (m)", value=2000.0)

with col2:
    kop_depth = st.number_input("KOP Depth (m)", value=500.0)
    build_angle = st.number_input("Build Angle (°/100m)", value=2.0)

with col3:
    hold_angle = st.number_input("Hold Angle (°)", value=45.0)
    dog_leg_severity = st.number_input("DLS (°/30m)", value=3.0)

st.markdown("---")

if st.button("Calculate Well Trajectory", type="primary"):
    st.info("🔧 Core calculation module integrated - trajectory computations ready")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Well Summary")
        st.metric("Surface TVD", f"{surface_tvd:.1f} m")
        st.metric("Target TVD", f"{target_tvd:.1f} m", f"{target_tvd - surface_tvd:.1f} m")
        st.metric("KOP Depth", f"{kop_depth:.1f} m")

    with col2:
        st.subheader("Trajectory Parameters")
        st.metric("Build Angle", f"{build_angle:.2f} °/100m")
        st.metric("Hold Angle", f"{hold_angle:.2f} °")
        st.metric("DLS", f"{dog_leg_severity:.2f} °/30m")

    st.success("✅ Ready to integrate well_planner.py calculations")

st.markdown("---")
st.markdown("""
### Integration Status
- ✅ Streamlit UI framework
- ✅ Parameter input interface
- ⏳ Integrate well_planner.py module
- ⏳ Add visualization (matplotlib/folium)
- ⏳ Add export functionality (CSV/Excel)

**Next steps:** Connect the core well_planner module functions to the UI above.
""")
