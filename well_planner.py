"""
Streamlit UI for Well Planner
Wraps well_planner.py functions and provides an interactive web interface.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Import from well_planner module
# (Assumes well_planner.py is in the same directory or installed as a module)
from well_planner import (
    design_trajectory, generate_stations, compute_survey, compute_uncertainty,
    plot_plan_and_section, plot_3d, export_csv, export_excel, _plot_map,
    list_survey_tools, SURVEY_TOOLS, DLS_INTERVAL, DEPTH_UNIT, DLS_UNIT,
    latlon_to_ne, ne_to_latlon
)

st.set_page_config(page_title="Well Planner", layout="wide", initial_sidebar_state="expanded")

st.title("🛢️ Directional Well Planner")
st.markdown("Interactive well trajectory design and analysis tool.")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Well info
    st.subheader("Well Information")
    well_name = st.text_input("Well Name", value="Well-1A")
    units = st.selectbox("Units", ["metric", "imperial"], index=0)
    dls_int = DLS_INTERVAL[units]
    depth_u = DEPTH_UNIT[units]
    dls_u = DLS_UNIT[units]

    # Location
    st.subheader("Location")
    coord_type = st.radio("Coordinates", ["Lat/Lon", "Northing/Easting"])

    crs_working = st.text_input("Working CRS (EPSG)", value="EPSG:32630",
                                help="e.g., EPSG:32630 for UTM zone 30N")

    if coord_type == "Lat/Lon":
        st.write("**Surface**")
        col1, col2 = st.columns(2)
        with col1:
            surface_lat = st.number_input("Surface Lat", value=57.6, format="%.6f")
        with col2:
            surface_lon = st.number_input("Surface Lon", value=1.85, format="%.6f")
        surface_n, surface_e = latlon_to_ne(surface_lat, surface_lon, crs_working)

        st.write("**Target**")
        col1, col2 = st.columns(2)
        with col1:
            target_lat = st.number_input("Target Lat", value=57.608, format="%.6f")
        with col2:
            target_lon = st.number_input("Target Lon", value=1.862, format="%.6f")
        target_n, target_e = latlon_to_ne(target_lat, target_lon, crs_working)
    else:
        st.write("**Surface**")
        col1, col2 = st.columns(2)
        with col1:
            surface_n = st.number_input(f"Surface Northing ({depth_u})", value=6387000.0)
        with col2:
            surface_e = st.number_input(f"Surface Easting ({depth_u})", value=206000.0)
        surface_lat, surface_lon = ne_to_latlon(surface_n, surface_e, crs_working)

        st.write("**Target**")
        col1, col2 = st.columns(2)
        with col1:
            target_n = st.number_input(f"Target Northing ({depth_u})", value=6388000.0)
        with col2:
            target_e = st.number_input(f"Target Easting ({depth_u})", value=207000.0)
        target_lat, target_lon = ne_to_latlon(target_n, target_e, crs_working)

    target_tvd = st.number_input(f"Target TVD ({depth_u})", value=2800.0, min_value=100.0)

    # Trajectory design
    st.subheader("Trajectory Design")
    kop_tvd = st.number_input(f"KOP TVD ({depth_u})", value=600.0, min_value=0.0)
    build_rate = st.number_input(f"Build Rate ({dls_u})", value=3.5, min_value=0.1)
    surface_dls = st.number_input(f"Surface DLS ({dls_u})", value=0.5, min_value=0.0,
                                  help="0 = vertical section before KOP")
    survey_interval = st.number_input(f"Survey Interval ({depth_u})", value=dls_int, min_value=1.0)

    # Survey tool selection
    st.subheader("Survey Tools")
    tool_section_count = st.number_input("Number of tool sections", value=2, min_value=1, max_value=5)

    tool_sections = []
    for i in range(tool_section_count):
        with st.expander(f"Section {i+1}"):
            from_md = st.number_input(f"From MD ({depth_u})", value=float(i*500), key=f"from_{i}")
            to_md_input = st.text_input(f"To MD ({depth_u}) or 'TD'", value="TD" if i == tool_section_count-1 else str((i+1)*500), key=f"to_{i}")
            to_md = None if to_md_input.upper() == "TD" else float(to_md_input)

            # Show available tools
            st.write("Available tools:")
            tool_codes = list(SURVEY_TOOLS.keys())
            tool_code = st.selectbox(f"Tool Code", tool_codes, key=f"tool_{i}")
            tool_info = SURVEY_TOOLS[tool_code]
            st.caption(tool_info['description'])

            tool_sections.append({
                'from_md': from_md,
                'to_md': to_md,
                'code': tool_code
            })

    confidence = st.number_input("Confidence (σ multiplier)", value=2.0, min_value=1.0, max_value=3.0,
                                help="1.0=68%, 2.0≈95%, 2.576≈99%")

# Main content
try:
    # Design trajectory
    traj = design_trajectory(
        surface_n, surface_e, target_n, target_e, target_tvd,
        kop_tvd=kop_tvd,
        build_rate=build_rate,
        surface_dls=surface_dls,
        units=units,
        survey_interval=survey_interval
    )

    # Generate stations and compute survey
    stations = generate_stations(traj, survey_interval, units)
    survey_df = compute_survey(stations, dls_int)

    # Compute uncertainty
    uncertainty_df = compute_uncertainty(survey_df, tool_sections, units, confidence)

    # Display summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Azimuth", f"{traj['azimuth']:.2f}°")
    with col2:
        st.metric("Max Inclination", f"{traj['inc_max']:.2f}°")
    with col3:
        st.metric("Horizontal Depth", f"{traj['H_total']:.1f} {depth_u}")
    with col4:
        st.metric("Total MD", f"{traj['md_eoh']:.1f} {depth_u}")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary", "📈 Plan & Section", "🎯 3D View", "📋 Survey Data", "⚠️ Uncertainty"])

    with tab1:
        st.subheader("Well Design Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Location**")
            st.write(f"Surface: {abs(surface_lat):.6f}° {'N' if surface_lat >= 0 else 'S'}, {abs(surface_lon):.6f}° {'E' if surface_lon >= 0 else 'W'}")
            st.write(f"Target: {abs(target_lat):.6f}° {'N' if target_lat >= 0 else 'S'}, {abs(target_lon):.6f}° {'E' if target_lon >= 0 else 'W'}")
            st.write(f"Target TVD: {target_tvd:.1f} {depth_u}")

        with col2:
            st.write("**Trajectory**")
            if surface_dls > 0:
                st.write(f"Surface DLS: {surface_dls:.2f} {dls_u}")
                st.write(f"Inc at KOP: {traj['inc_at_kop']:.2f}°")
            st.write(f"KOP: {traj['kop_tvd']:.1f} {depth_u} TVD / {traj['md_kop']:.1f} {depth_u} MD")
            st.write(f"Build Rate: {build_rate:.2f} {dls_u}")
            st.write(f"EOB MD: {traj['md_eob']:.1f} {depth_u}")
            st.write(f"Hold MD: {traj['md_eoh']:.1f} {depth_u}")

        st.write("**Survey Tool Sections**")
        for i, sec in enumerate(tool_sections):
            to = f"{sec['to_md']:.0f} {depth_u}" if sec['to_md'] else 'TD'
            tool_name = SURVEY_TOOLS[sec['code']]['name']
            st.write(f"{i+1}. {sec['from_md']:.0f} → {to} {depth_u}: **{sec['code']}** — {tool_name}")

        st.write("**Position Uncertainty at TD (~95% confidence)**")
        td_unc = uncertainty_df.iloc[-1]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lateral", f"±{td_unc['σ_lat']:.1f} {depth_u}")
        with col2:
            st.metric("High-Side", f"±{td_unc['σ_hs']:.1f} {depth_u}")
        with col3:
            st.metric("TVD", f"±{td_unc['σ_tvd']:.1f} {depth_u}")

    with tab2:
        st.subheader("Plan View & Vertical Section")
        fig = plot_plan_and_section(
            survey_df, traj, surface_n, surface_e,
            target_n, target_e, units, well_name=well_name,
            uncertainty_df=uncertainty_df
        )
        st.pyplot(fig)

    with tab3:
        st.subheader("3D Trajectory")
        fig3d = plot_3d(survey_df, traj, units, well_name=well_name)
        st.pyplot(fig3d)

    with tab4:
        st.subheader("Survey Station Data")

        # Show full dataframe
        st.write(uncertainty_df.round(4))

        # Download options
        col1, col2 = st.columns(2)

        with col1:
            # CSV export
            csv_data = uncertainty_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"{well_name}_survey.csv",
                mime="text/csv"
            )

        with col2:
            # Excel export (using openpyxl)
            try:
                from io import BytesIO
                import openpyxl
                from openpyxl.utils.dataframe import dataframe_to_rows

                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    uncertainty_df.to_excel(writer, sheet_name='Survey', index=False)

                    summary = {
                        'Parameter': [
                            'Units', 'Azimuth (deg)', f'KOP TVD ({depth_u})',
                            f'KOP MD ({depth_u})', 'Inc at KOP (deg)',
                            f'Build Rate ({dls_u})', 'Max Inclination (deg)',
                            f'EOB MD ({depth_u})', f'Hold MD ({depth_u})',
                            f'TD MD ({depth_u})', f'H_total ({depth_u})',
                            f'Target TVD ({depth_u})'
                        ],
                        'Value': [
                            units, f"{traj['azimuth']:.2f}", f"{traj['kop_tvd']:.2f}",
                            f"{traj['md_kop']:.2f}", f"{traj['inc_at_kop']:.2f}",
                            f"{build_rate:.2f}", f"{traj['inc_max']:.2f}",
                            f"{traj['md_eob']:.2f}", f"{traj['md_eoh']:.2f}",
                            f"{traj['md_eoh']:.2f}", f"{traj['H_total']:.2f}",
                            f"{target_tvd:.2f}"
                        ]
                    }
                    pd.DataFrame(summary).to_excel(writer, sheet_name='Design Summary', index=False)

                output.seek(0)
                st.download_button(
                    label="📥 Download Excel",
                    data=output.getvalue(),
                    file_name=f"{well_name}_survey.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.warning(f"Excel export not available: {e}")

    with tab5:
        st.subheader("Position Uncertainty Analysis")

        # Plot uncertainty vs MD
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(uncertainty_df['MD'], uncertainty_df['σ_lat'], 'b-', label='Lateral', linewidth=2)
        ax1.plot(uncertainty_df['MD'], uncertainty_df['σ_hs'], 'r-', label='High-Side', linewidth=2)
        ax1.plot(uncertainty_df['MD'], uncertainty_df['σ_tvd'], 'g-', label='TVD', linewidth=2)
        ax1.set_xlabel(f'Measured Depth ({depth_u})')
        ax1.set_ylabel(f'Uncertainty (±{depth_u})')
        ax1.set_title('Position Uncertainty vs Measured Depth')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(uncertainty_df['MD'], uncertainty_df['σ_H'], 'purple', linewidth=2, label='Horizontal (H)')
        ax2.fill_between(uncertainty_df['MD'], 0, uncertainty_df['σ_H'], alpha=0.2, color='purple')
        ax2.set_xlabel(f'Measured Depth ({depth_u})')
        ax2.set_ylabel(f'Uncertainty (±{depth_u})')
        ax2.set_title('Horizontal Position Uncertainty')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

        # Uncertainty table
        st.write("**Uncertainty at Key Stations**")
        key_indices = [0, len(uncertainty_df)//4, len(uncertainty_df)//2, 3*len(uncertainty_df)//4, -1]
        key_df = uncertainty_df.iloc[key_indices][['MD', 'Inc', 'Azi', 'TVD', 'σ_lat', 'σ_hs', 'σ_tvd', 'σ_H']].round(2)
        st.dataframe(key_df)

except Exception as e:
    st.error(f"❌ Error in calculation: {str(e)}")
    st.info("Check your input parameters and ensure they are valid.")

st.divider()
st.caption("🛢️ Directional Well Planner | Well design and trajectory analysis")
