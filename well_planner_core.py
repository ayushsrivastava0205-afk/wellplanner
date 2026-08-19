"""
well_planner.py
===============
Core module for directional well planning.

Functions cover:
  - Coordinate conversion (lat/lon <-> northing/easting via pyproj)
  - Minimum curvature survey computation
  - Trajectory design (KOP, build, hold sections)
  - Plan-view, vertical-section, and 3-D plotting
  - Interactive folium map export
  - CSV / Excel export
  - High-level run_well_plan() entry point
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pyproj import Transformer
import folium

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

DLS_INTERVAL = {'metric': 30.0, 'imperial': 100.0}
DEPTH_UNIT   = {'metric': 'm', 'imperial': 'ft'}
DLS_UNIT     = {'metric': 'deg/30m', 'imperial': 'deg/100ft'}

# ---------------------------------------------------------------------------
# SURVEY TOOL DEFINITIONS  (ISCWSA-simplified accuracy models)
# ---------------------------------------------------------------------------

SURVEY_TOOLS = {
    'BLIND': {
        'name':         'Blind (No Survey)',
        'has_inc':      False,
        'has_azi':      False,
        'inc_accuracy': None,     # 1σ degrees
        'azi_accuracy': None,     # 1σ degrees
        'depth_sf':     0.0035,   # depth scale factor ΔD/D
        'azimuth_ref':  None,
        'description':  'No survey tool. Well assumed to continue in current direction.',
    },
    'INC_ONLY': {
        'name':         'Inclination Only',
        'has_inc':      True,
        'has_azi':      False,
        'inc_accuracy': 0.50,
        'azi_accuracy': None,
        'depth_sf':     0.0035,
        'azimuth_ref':  None,
        'description':  'Accelerometer only. Azimuth held from last known value; lateral uncertainty grows as random walk.',
    },
    'SINGLE_SHOT': {
        'name':         'Magnetic Single Shot (MSS)',
        'has_inc':      True,
        'has_azi':      True,
        'inc_accuracy': 0.25,
        'azi_accuracy': 1.50,
        'depth_sf':     0.0035,
        'azimuth_ref':  'magnetic',
        'description':  'Point magnetic survey. Standard for conventional directional wells.',
    },
    'MULTI_SHOT': {
        'name':         'Magnetic Multi Shot (MMS)',
        'has_inc':      True,
        'has_azi':      True,
        'inc_accuracy': 0.25,
        'azi_accuracy': 1.50,
        'depth_sf':     0.0035,
        'azimuth_ref':  'magnetic',
        'description':  'Records multiple stations per wireline run. Same accuracy as single shot.',
    },
    'MWD': {
        'name':         'MWD (Standard)',
        'has_inc':      True,
        'has_azi':      True,
        'inc_accuracy': 0.10,
        'azi_accuracy': 0.50,
        'depth_sf':     0.0035,
        'azimuth_ref':  'magnetic',
        'description':  'Measurement While Drilling. Industry standard for directional wells.',
    },
    'MWD_IFR': {
        'name':         'MWD + IFR',
        'has_inc':      True,
        'has_azi':      True,
        'inc_accuracy': 0.10,
        'azi_accuracy': 0.30,
        'depth_sf':     0.0035,
        'azimuth_ref':  'corrected',
        'description':  'MWD with In-Field Referencing. Corrects local magnetic field variations.',
    },
    'MWD_IFR_MS': {
        'name':         'MWD + IFR + Multi-Station',
        'has_inc':      True,
        'has_azi':      True,
        'inc_accuracy': 0.10,
        'azi_accuracy': 0.20,
        'depth_sf':     0.0035,
        'azimuth_ref':  'corrected',
        'description':  'MWD with IFR and multi-station analysis. Industry best-practice magnetic survey.',
    },
    'GYRO_SS': {
        'name':         'Gyroscopic Single Shot',
        'has_inc':      True,
        'has_azi':      True,
        'inc_accuracy': 0.10,
        'azi_accuracy': 0.50,
        'depth_sf':     0.0035,
        'azimuth_ref':  'true_north',
        'description':  'Gyro single shot. Unaffected by magnetic interference. Used in cased hole.',
    },
    'GYRO_CONT': {
        'name':         'Continuous Gyro',
        'has_inc':      True,
        'has_azi':      True,
        'inc_accuracy': 0.05,
        'azi_accuracy': 0.10,
        'depth_sf':     0.0035,
        'azimuth_ref':  'true_north',
        'description':  'Highest accuracy available. Used near infrastructure or when magnetics unreliable.',
    },
}

# ---------------------------------------------------------------------------
# Common EPSG codes for reference
# ---------------------------------------------------------------------------
# EPSG:32629  WGS84 / UTM zone 29N
# EPSG:32630  WGS84 / UTM zone 30N   (central North Sea)
# EPSG:32631  WGS84 / UTM zone 31N
# EPSG:32632  WGS84 / UTM zone 32N   (Norwegian sector)
# EPSG:23029  ED50  / UTM zone 29N
# EPSG:23030  ED50  / UTM zone 30N
# EPSG:23031  ED50  / UTM zone 31N
# EPSG:23032  ED50  / UTM zone 32N
# EPSG:26729  NAD27 / UTM zone 29N
# EPSG:26730  NAD27 / UTM zone 30N
# EPSG:4326   WGS84 geographic (lat/lon)
# EPSG:4230   ED50  geographic
# EPSG:4267   NAD27 geographic

# ---------------------------------------------------------------------------
# COORDINATE CONVERSION
# ---------------------------------------------------------------------------

def latlon_to_ne(lat, lon, crs_to):
    """
    Convert geographic coordinates to projected northing/easting.

    Parameters
    ----------
    lat : float  — latitude in decimal degrees
    lon : float  — longitude in decimal degrees
    crs_to : str/int — target CRS (e.g. 'EPSG:32630')

    Returns
    -------
    (northing, easting) in the target CRS units (usually metres)
    """
    transformer = Transformer.from_crs('EPSG:4326', crs_to, always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return northing, easting


def ne_to_latlon(northing, easting, crs_from):
    """
    Convert projected northing/easting back to geographic lat/lon.

    Parameters
    ----------
    northing : float
    easting  : float
    crs_from : str/int — source CRS (e.g. 'EPSG:32630')

    Returns
    -------
    (lat, lon) in decimal degrees (WGS84)
    """
    transformer = Transformer.from_crs(crs_from, 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lat, lon


# ---------------------------------------------------------------------------
# MINIMUM CURVATURE
# ---------------------------------------------------------------------------

def dogleg_angle(inc1, azi1, inc2, azi2):
    """
    Compute dogleg angle between two survey stations.

    Uses the numerically stable spherical law of cosines form:
      DL = 2 * arcsin( sqrt( sin²(dI/2) + sin(I1)*sin(I2)*sin²(dA/2) ) )

    Parameters
    ----------
    inc1, azi1 : float — inclination and azimuth at station 1 (degrees)
    inc2, azi2 : float — inclination and azimuth at station 2 (degrees)

    Returns
    -------
    dogleg : float — dogleg angle in radians
    """
    i1 = math.radians(inc1)
    i2 = math.radians(inc2)
    a1 = math.radians(azi1)
    a2 = math.radians(azi2)

    di = (i2 - i1) / 2.0
    da = (a2 - a1) / 2.0

    arg = math.sin(di)**2 + math.sin(i1) * math.sin(i2) * math.sin(da)**2
    arg = max(0.0, min(1.0, arg))          # clip for numerical safety
    return 2.0 * math.asin(math.sqrt(arg))


def min_curvature_step(md1, inc1, azi1, md2, inc2, azi2, dls_interval):
    """
    Minimum-curvature calculation for a single survey interval.

    Parameters
    ----------
    md1, md2          : float — measured depths
    inc1, inc2        : float — inclinations (degrees)
    azi1, azi2        : float — azimuths (degrees)
    dls_interval      : float — reference length for DLS (30 m or 100 ft)

    Returns
    -------
    (dTVD, dN, dE, dogleg_deg, dls)
      dTVD  : TVD increment
      dN    : northing increment
      dE    : easting increment
      dogleg_deg : dogleg angle for this interval (degrees)
      dls   : dogleg severity (degrees per dls_interval)
    """
    dmd = md2 - md1
    dl  = dogleg_angle(inc1, azi1, inc2, azi2)   # radians

    dogleg_deg = math.degrees(dl)
    dls = (dogleg_deg / dmd * dls_interval) if dmd > 1e-12 else 0.0

    # Ratio factor
    if dl < 1e-10:
        rf = 1.0
    else:
        rf = 2.0 / dl * math.tan(dl / 2.0)

    i1 = math.radians(inc1)
    i2 = math.radians(inc2)
    a1 = math.radians(azi1)
    a2 = math.radians(azi2)

    half_dmd = dmd / 2.0

    dTVD = half_dmd * (math.cos(i1) + math.cos(i2)) * rf
    dN   = half_dmd * (math.sin(i1) * math.cos(a1) + math.sin(i2) * math.cos(a2)) * rf
    dE   = half_dmd * (math.sin(i1) * math.sin(a1) + math.sin(i2) * math.sin(a2)) * rf

    return dTVD, dN, dE, dogleg_deg, dls


def compute_survey(stations, dls_interval):
    """
    Build a full survey table using minimum curvature.

    Parameters
    ----------
    stations     : list of (md, inc, azi) tuples
    dls_interval : float — 30 (metric) or 100 (imperial)

    Returns
    -------
    DataFrame with columns:
      MD, Inc, Azi, TVD, N, E, DLS, Dogleg, VS, Closure_Azi
    """
    rows = []
    tvd = n = e = 0.0

    # first station
    md0, inc0, azi0 = stations[0]
    rows.append({
        'MD': md0, 'Inc': inc0, 'Azi': azi0,
        'TVD': tvd, 'N': n, 'E': e,
        'DLS': 0.0, 'Dogleg': 0.0,
        'VS': 0.0, 'Closure_Azi': 0.0
    })

    for md1, inc1, azi1 in stations[1:]:
        md0, inc0, azi0 = rows[-1]['MD'], rows[-1]['Inc'], rows[-1]['Azi']
        dTVD, dN, dE, dogleg_deg, dls = min_curvature_step(
            md0, inc0, azi0, md1, inc1, azi1, dls_interval)
        tvd += dTVD
        n   += dN
        e   += dE
        vs  = math.hypot(n, e)
        cl_azi = math.degrees(math.atan2(e, n)) % 360.0
        rows.append({
            'MD': md1, 'Inc': inc1, 'Azi': azi1,
            'TVD': tvd, 'N': n, 'E': e,
            'DLS': dls, 'Dogleg': dogleg_deg,
            'VS': vs, 'Closure_Azi': cl_azi
        })

    df = pd.DataFrame(rows)
    return df


def list_survey_tools():
    """Print a formatted table of all available survey tool codes and accuracy specs."""
    header = f"\n  {'Code':<15} {'Name':<35} {'Inc 1σ (°)':<13} {'Azi 1σ (°)':<13} Ref"
    print(header)
    print('  ' + '-' * 80)
    for code, t in SURVEY_TOOLS.items():
        inc = f"{t['inc_accuracy']:.2f}" if t['inc_accuracy'] is not None else 'N/A'
        azi = f"{t['azi_accuracy']:.2f}" if t['azi_accuracy'] is not None else 'N/A'
        ref = t['azimuth_ref'] or 'N/A'
        print(f"  {code:<15} {t['name']:<35} {inc:<13} {azi:<13} {ref}")
    print()


def _get_tool_at_md(md, tool_sections):
    """
    Return (tool_dict, code) for the survey tool active at a given MD.

    tool_sections is a list of dicts:
      [{'from_md': 0, 'to_md': 600, 'code': 'INC_ONLY'},
       {'from_md': 600, 'to_md': None, 'code': 'MWD_IFR_MS'}]
    'to_md': None means open-ended (to TD).
    """
    for sec in tool_sections:
        to_md = sec.get('to_md')
        if md >= sec['from_md'] and (to_md is None or md <= to_md):
            code = sec['code']
            if code not in SURVEY_TOOLS:
                raise ValueError(
                    f"Unknown survey tool code {code!r}. "
                    f"Valid codes: {list(SURVEY_TOOLS)}"
                )
            return SURVEY_TOOLS[code], code
    # fallback: last section's tool
    code = tool_sections[-1]['code']
    return SURVEY_TOOLS[code], code


def _uncertainty_corridor(xs, ys, sigmas):
    """
    Build upper/lower polygon bounds for a shaded uncertainty corridor around a path.

    xs, ys   : lists of path coordinates
    sigmas   : list of uncertainty half-widths (same length as xs/ys)

    Returns (upper_xs, upper_ys, lower_xs, lower_ys) — each a list.
    """
    upper_x, upper_y = [], []
    lower_x, lower_y = [], []

    n = len(xs)
    for i in range(n):
        if i < n - 1:
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
        else:
            dx = xs[i] - xs[i - 1]
            dy = ys[i] - ys[i - 1]

        length = math.hypot(dx, dy)
        if length > 1e-10:
            nx, ny = -dy / length, dx / length   # left-hand perpendicular
        else:
            nx, ny = 1.0, 0.0

        s = sigmas[i]
        upper_x.append(xs[i] + nx * s)
        upper_y.append(ys[i] + ny * s)
        lower_x.append(xs[i] - nx * s)
        lower_y.append(ys[i] - ny * s)

    return upper_x, upper_y, lower_x, lower_y


def compute_uncertainty(survey_df, tool_sections, units='metric', confidence=2.0):
    """
    Compute position uncertainty at each survey station using a simplified
    systematic-error propagation model (inspired by ISCWSA).

    Error sources modelled:
      - Inclination error  → high-side and TVD uncertainty
      - Azimuth error      → lateral uncertainty
      - Depth scale factor → along-hole depth uncertainty
      - INC_ONLY/BLIND     → azimuth treated as random walk (1 deg/30 m drift)

    Parameters
    ----------
    survey_df      : DataFrame from compute_survey()
    tool_sections  : list of {'from_md', 'to_md'|None, 'code'} dicts
    units          : 'metric' or 'imperial'
    confidence     : multiplier  (1.0 = 1σ ≈ 68%,  2.0 ≈ 95%,  2.576 ≈ 99%)

    Returns
    -------
    DataFrame — copy of survey_df with added columns:
      Tool_Code, Tool_Name, σ_lat (m), σ_hs (m), σ_tvd (m), σ_H (m)
    """
    di = DLS_INTERVAL[units]     # 30 m or 100 ft — used for random-walk scaling

    sigma_lat, sigma_hs, sigma_tvd = [], [], []
    tool_codes_col, tool_names_col = [], []

    for _, row in survey_df.iterrows():
        md  = row['MD']
        inc = math.radians(row['Inc'])

        tool, code = _get_tool_at_md(md, tool_sections)
        tool_codes_col.append(code)
        tool_names_col.append(tool['name'])

        if md < 1e-6:
            sigma_lat.append(0.0)
            sigma_hs.append(0.0)
            sigma_tvd.append(0.0)
            continue

        inc_acc_rad = math.radians(tool['inc_accuracy']) if tool['inc_accuracy'] else 0.0
        depth_sf    = tool['depth_sf']

        if tool['has_azi'] and tool['azi_accuracy'] is not None:
            azi_acc_rad = math.radians(tool['azi_accuracy'])
        else:
            # No azimuth measurement: model azimuth as random walk
            # (1 deg per survey_interval drift, accumulated as sqrt(n_stations))
            drift_per_interval = math.radians(1.0)
            n_intervals = md / di
            azi_acc_rad = drift_per_interval * math.sqrt(max(n_intervals, 1))

        # Propagated 1σ uncertainties (systematic model):
        #   lateral  = azi_error × MD × sin(Inc)
        #   high-side = inc_error × MD
        #   TVD       = inc_error × MD × sin(Inc) + depth_SF × TVD
        sl = confidence * azi_acc_rad * md * math.sin(inc)
        sh = confidence * inc_acc_rad * md
        st = confidence * (inc_acc_rad * md * abs(math.sin(inc))
                           + depth_sf * row['TVD'])

        sigma_lat.append(sl)
        sigma_hs.append(sh)
        sigma_tvd.append(st)

    df = survey_df.copy()
    df.insert(df.columns.get_loc('DLS'), 'Tool_Code', tool_codes_col)
    df.insert(df.columns.get_loc('DLS'), 'Tool_Name', tool_names_col)
    df['σ_lat'] = sigma_lat
    df['σ_hs']  = sigma_hs
    df['σ_tvd'] = sigma_tvd
    df['σ_H']   = [math.hypot(l, h) for l, h in zip(sigma_lat, sigma_hs)]
    return df


# ---------------------------------------------------------------------------
# TRAJECTORY DESIGN
# ---------------------------------------------------------------------------

def design_trajectory(surface_n, surface_e, target_n, target_e, target_tvd,
                      kop_tvd, build_rate, surface_dls=0.0,
                      units='metric', survey_interval=None):
    """
    Design a build-and-hold directional well trajectory.

    Parameters
    ----------
    surface_n/e   : float — surface northing/easting
    target_n/e    : float — target northing/easting
    target_tvd    : float — target TVD
    kop_tvd       : float — kick-off point TVD
    build_rate    : float — build rate in deg/dls_interval
    surface_dls   : float — pre-KOP surface DLS (0 = vertical)
    units         : 'metric' or 'imperial'
    survey_interval : float (optional) — not used here, passed through

    Returns
    -------
    dict with all trajectory parameters
    """
    dls_int = DLS_INTERVAL[units]

    dN = target_n - surface_n
    dE = target_e - surface_e
    H_total = math.hypot(dN, dE)
    azimuth = math.degrees(math.atan2(dE, dN)) % 360.0

    if target_tvd <= kop_tvd:
        raise ValueError("Target TVD must be deeper than KOP")

    # Build radius
    R = dls_int / math.radians(build_rate)

    # Surface section (pre-KOP)
    if surface_dls > 1e-10:
        R_surf = dls_int / math.radians(surface_dls)
        # Inclination at KOP via circular arc geometry
        # TVD gained on circular arc of radius R_surf from vertical:
        # TVD = R_surf * sin(inc_at_kop)  →  inc_at_kop = arcsin(kop_tvd / R_surf)
        ratio = kop_tvd / R_surf
        ratio = min(ratio, 0.999)
        inc_at_kop_rad = math.asin(ratio)
        H_at_kop = R_surf * (1.0 - math.cos(inc_at_kop_rad))
        # MD from surface to KOP = arc length
        md_kop = R_surf * inc_at_kop_rad
    else:
        R_surf = None
        inc_at_kop_rad = 0.0
        H_at_kop = 0.0
        md_kop = kop_tvd        # vertical section, MD == TVD

    inc_at_kop_deg = math.degrees(inc_at_kop_rad)
    theta0 = inc_at_kop_rad

    # Effective horizontal and TVD distances from KOP to target
    H_eff    = H_total - H_at_kop
    dTVD_eff = target_tvd - kop_tvd

    if dTVD_eff <= 0:
        raise ValueError("Target TVD must be deeper than KOP")

    # Solve for max inclination (theta) numerically
    # Equation: R*(1-cos(theta-theta0)) + H_eff*cos(theta) - dTVD_eff*sin(theta) = 0
    # (derived from: TVD_build + TVD_hold = dTVD_eff,  H_build + H_hold = H_eff)
    # TVD_build = R*sin(theta) - R*sin(theta0)
    # H_build   = R*(cos(theta0) - cos(theta))
    # H_hold    = H_eff - H_build
    # TVD_hold  = H_hold / tan(theta) = H_hold * cos(theta) / sin(theta)
    # TVD_build + TVD_hold = dTVD_eff
    # R*(sin(theta)-sin(theta0)) + [H_eff - R*(cos(theta0)-cos(theta))]*cos(theta)/sin(theta) = dTVD_eff
    # Multiply through by sin(theta):
    # R*(sin(theta)-sin(theta0))*sin(theta) + [H_eff - R*(cos(theta0)-cos(theta))]*cos(theta) = dTVD_eff*sin(theta)
    # Rearranging to f(theta):
    # R*(1-cos(theta-theta0)) + H_eff*cos(theta) - dTVD_eff*sin(theta) = 0
    # (using trig identity: sin(t)(cos(t)-sin(t0)*sin-...) simplification lands here)

    def f(theta):
        return (R * (1.0 - math.cos(theta - theta0))
                + H_eff * math.cos(theta)
                - dTVD_eff * math.sin(theta))

    # Scan 0..89 deg to find a sign change bracket
    thetas = np.linspace(math.radians(0.01), math.radians(89.0), 890)
    fvals  = [f(t) for t in thetas]
    bracket = None
    for k in range(len(fvals) - 1):
        if fvals[k] * fvals[k + 1] < 0:
            bracket = (thetas[k], thetas[k + 1])
            break

    if bracket is None:
        raise ValueError(
            "Could not find a valid inclination solution. "
            "Try increasing the build rate, adjusting the KOP, "
            "or verifying that the target is reachable with the given parameters."
        )

    inc_max_rad = brentq(f, bracket[0], bracket[1], xtol=1e-10, rtol=1e-12)
    inc_max_deg = math.degrees(inc_max_rad)

    # Section lengths (measured depth)
    build_md = (inc_max_deg - inc_at_kop_deg) / build_rate * dls_int
    H_build  = R * (math.cos(theta0) - math.cos(inc_max_rad))
    H_hold   = H_eff - H_build
    if abs(math.sin(inc_max_rad)) < 1e-10:
        hold_md = 0.0
    else:
        hold_md = H_hold / math.sin(inc_max_rad)

    md_eob = md_kop + build_md
    md_eoh = md_eob + hold_md

    return {
        'azimuth':       azimuth,
        'inc_at_kop':    inc_at_kop_deg,
        'inc_max':       inc_max_deg,
        'build_rate':    build_rate,
        'surface_dls':   surface_dls,
        'build_radius':  R,
        'kop_tvd':       kop_tvd,
        'md_kop':        md_kop,
        'build_md':      build_md,
        'md_eob':        md_eob,
        'hold_md':       hold_md,
        'md_eoh':        md_eoh,
        'H_total':       H_total,
        'H_at_kop':      H_at_kop,
        'H_eff':         H_eff,
        'dN':            dN,
        'dE':            dE,
        'target_tvd':    target_tvd,
        'units':         units,
        'R_surf':        R_surf,
        'inc_at_kop_rad': inc_at_kop_rad,
    }


def generate_stations(traj, survey_interval, units):
    """
    Generate (md, inc, azi) survey station tuples for the designed trajectory.

    Sections:
      0 → KOP  : inclination built along surface circular arc (if surface_dls>0)
      KOP → EOB: build section at build_rate
      EOB → EOH: hold section at inc_max

    Parameters
    ----------
    traj            : dict returned by design_trajectory()
    survey_interval : float — spacing between survey stations
    units           : 'metric' or 'imperial'

    Returns
    -------
    list of (md, inc, azi) tuples, deduplicated by MD
    """
    dls_int     = DLS_INTERVAL[units]
    azimuth     = traj['azimuth']
    md_kop      = traj['md_kop']
    md_eob      = traj['md_eob']
    md_eoh      = traj['md_eoh']
    inc_at_kop  = traj['inc_at_kop']
    inc_max     = traj['inc_max']
    build_rate  = traj['build_rate']
    surface_dls = traj['surface_dls']
    R_surf      = traj['R_surf']

    raw = []

    # -----------------------------------------------------------------------
    # Surface → KOP
    # -----------------------------------------------------------------------
    mds_surf = list(np.arange(0.0, md_kop, survey_interval))
    for md in mds_surf:
        if surface_dls > 1e-10 and R_surf is not None:
            # Circular arc: TVD = R_surf * sin(inc), MD = R_surf * inc
            # So inc = arcsin(md / R_surf)
            inc_rad = math.asin(min(md / R_surf, 0.999))
            inc_deg = math.degrees(inc_rad)
            inc_deg = min(inc_deg, inc_at_kop)
        else:
            inc_deg = 0.0
        raw.append((md, inc_deg, azimuth))

    # Exact KOP point
    raw.append((md_kop, inc_at_kop, azimuth))

    # -----------------------------------------------------------------------
    # KOP → EOB  (build section)
    # -----------------------------------------------------------------------
    mds_build = list(np.arange(md_kop, md_eob, survey_interval))
    for md in mds_build:
        if md <= md_kop:
            continue
        inc_deg = inc_at_kop + build_rate * (md - md_kop) / dls_int
        inc_deg = min(inc_deg, inc_max)
        raw.append((md, inc_deg, azimuth))

    # Exact EOB point
    raw.append((md_eob, inc_max, azimuth))

    # -----------------------------------------------------------------------
    # EOB → EOH  (hold section)
    # -----------------------------------------------------------------------
    mds_hold = list(np.arange(md_eob, md_eoh, survey_interval))
    for md in mds_hold:
        if md <= md_eob:
            continue
        raw.append((md, inc_max, azimuth))

    # Exact EOH / TD point
    raw.append((md_eoh, inc_max, azimuth))

    # -----------------------------------------------------------------------
    # Deduplicate by MD (tolerance 0.01)
    # -----------------------------------------------------------------------
    raw.sort(key=lambda x: x[0])
    deduped = [raw[0]]
    for station in raw[1:]:
        if abs(station[0] - deduped[-1][0]) > 0.01:
            deduped.append(station)

    return deduped


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------

def plot_plan_and_section(survey_df, traj, surface_n, surface_e,
                          target_n, target_e, units, well_name='Well',
                          uncertainty_df=None):
    """
    Create a 1×2 figure: plan view (left) and vertical section (right).

    Returns
    -------
    matplotlib.figure.Figure
    """
    depth_u = DEPTH_UNIT[units]
    dls_u   = DLS_UNIT[units]

    # Locate key stations
    kop_row = survey_df.iloc[(survey_df['TVD'] - traj['kop_tvd']).abs().argsort()[:1]]
    eob_row = survey_df.iloc[(survey_df['MD'] - traj['md_eob']).abs().argsort()[:1]]

    fig, (ax_plan, ax_vs) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle(f"Well Plan — {well_name}  [{units}]", fontsize=14, fontweight='bold')

    # -------------------------------------------------------------------
    # LEFT: Plan view  (E on x-axis, N on y-axis)
    # -------------------------------------------------------------------
    ax_plan.plot(survey_df['E'], survey_df['N'],
                 'b-', linewidth=1.8, label='Wellpath', zorder=3)

    ax_plan.plot(0, 0, 'g^', markersize=12,
                 label='Surface', zorder=5)
    ax_plan.plot(target_e - surface_e, target_n - surface_n,
                 'r*', markersize=14, label='Target', zorder=5)
    ax_plan.plot(kop_row['E'].values[0], kop_row['N'].values[0],
                 'ko', markersize=9, label='KOP', zorder=5)
    ax_plan.plot(eob_row['E'].values[0], eob_row['N'].values[0],
                 'ms', markersize=9, label='EOB', zorder=5)

    # Uncertainty corridor — plan view
    if uncertainty_df is not None and uncertainty_df['σ_H'].max() > 0:
        ux, uy, lx, ly = _uncertainty_corridor(
            list(survey_df['E']), list(survey_df['N']),
            list(uncertainty_df['σ_H'])
        )
        poly_x = ux + list(reversed(lx))
        poly_y = uy + list(reversed(ly))
        ax_plan.fill(poly_x, poly_y, alpha=0.15, color='cyan', label='Uncertainty corridor')
        ax_plan.plot(ux, uy, 'c--', lw=0.8, alpha=0.6)
        ax_plan.plot(lx, ly, 'c--', lw=0.8, alpha=0.6)

    ax_plan.set_xlabel(f'Easting ({depth_u})')
    ax_plan.set_ylabel(f'Northing ({depth_u})')
    ax_plan.set_title('Plan View')
    ax_plan.set_aspect('equal', adjustable='datalim')
    ax_plan.grid(True, linestyle='--', alpha=0.5)
    ax_plan.legend(loc='best')

    # -------------------------------------------------------------------
    # RIGHT: Vertical section  (VS on x-axis, TVD on y-axis, inverted)
    # -------------------------------------------------------------------
    ax_vs.plot(survey_df['VS'], survey_df['TVD'],
               'b-', linewidth=1.8, label='Wellpath', zorder=3)

    ax_vs.plot(0, 0, 'g^', markersize=12,
               label='Surface', zorder=5)
    ax_vs.plot(math.hypot(target_n - surface_n, target_e - surface_e),
               traj['target_tvd'],
               'r*', markersize=14, label='Target', zorder=5)
    ax_vs.plot(kop_row['VS'].values[0], kop_row['TVD'].values[0],
               'ko', markersize=9, label='KOP', zorder=5)
    ax_vs.plot(eob_row['VS'].values[0], eob_row['TVD'].values[0],
               'ms', markersize=9, label='EOB', zorder=5)

    # Uncertainty corridor — section view (σ_hs in horizontal, σ_tvd in vertical)
    if uncertainty_df is not None and uncertainty_df['σ_hs'].max() > 0:
        vs_vals  = list(survey_df['VS'])
        tvd_vals = list(survey_df['TVD'])
        # lateral width = σ_hs (perpendicular to wellbore in section plane)
        ux, uy, lx, ly = _uncertainty_corridor(
            vs_vals, tvd_vals, list(uncertainty_df['σ_hs'])
        )
        poly_x = ux + list(reversed(lx))
        poly_y = uy + list(reversed(ly))
        ax_vs.fill(poly_x, poly_y, alpha=0.15, color='cyan', label='Uncertainty corridor')
        ax_vs.plot(ux, uy, 'c--', lw=0.8, alpha=0.6)
        ax_vs.plot(lx, ly, 'c--', lw=0.8, alpha=0.6)

    ax_vs.invert_yaxis()
    ax_vs.set_xlabel(f'Vertical Section ({depth_u})')
    ax_vs.set_ylabel(f'TVD ({depth_u})')
    ax_vs.set_title('Vertical Section')
    ax_vs.set_aspect('equal', adjustable='datalim')
    ax_vs.grid(True, linestyle='--', alpha=0.5)
    ax_vs.legend(loc='best')

    # Annotation: max inclination and build rate
    ax_vs.annotate(
        f"Max Inc: {traj['inc_max']:.2f}°\nBuild: {traj['build_rate']:.2f} {dls_u}",
        xy=(eob_row['VS'].values[0], eob_row['TVD'].values[0]),
        xytext=(eob_row['VS'].values[0] + 20, eob_row['TVD'].values[0] - 50),
        arrowprops=dict(arrowstyle='->', color='magenta'),
        fontsize=10, color='magenta',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    # TD uncertainty text
    if uncertainty_df is not None:
        td = uncertainty_df.iloc[-1]
        conf_pct = '~95%'   # shown as fixed label; user sets confidence multiplier
        ax_vs.text(
            0.02, 0.02,
            f"TD Uncertainty ({conf_pct}):\n"
            f"  Lateral: ±{td['σ_lat']:.1f} {depth_u}\n"
            f"  High-side: ±{td['σ_hs']:.1f} {depth_u}\n"
            f"  TVD: ±{td['σ_tvd']:.1f} {depth_u}",
            transform=ax_vs.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        )

    plt.tight_layout()
    return fig


def plot_3d(survey_df, traj, units, well_name='Well'):
    """
    3-D trajectory plot with E, N, TVD axes (TVD increases downward).

    Returns
    -------
    matplotlib.figure.Figure
    """
    depth_u = DEPTH_UNIT[units]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    ax.plot(survey_df['E'], survey_df['N'], -survey_df['TVD'],
            'b-', linewidth=1.8, label='Wellpath')

    ax.scatter([0], [0], [0], color='green', s=100, marker='^',
               label='Surface', zorder=5)
    ax.scatter([survey_df['E'].iloc[-1]], [survey_df['N'].iloc[-1]],
               [-survey_df['TVD'].iloc[-1]],
               color='red', s=150, marker='*', label='Target', zorder=5)

    ax.set_xlabel(f'Easting ({depth_u})')
    ax.set_ylabel(f'Northing ({depth_u})')
    ax.set_zlabel(f'TVD ({depth_u}) — down')
    ax.set_title(f'3-D Trajectory — {well_name}  [{units}]')
    ax.legend()

    plt.tight_layout()
    return fig


def plot_map(survey_df, surface_lat, surface_lon, target_lat, target_lon,
             surface_n, surface_e, crs_working, units, save_path):
    """
    Build and save an interactive folium map of the well trajectory.

    Parameters
    ----------
    survey_df   : DataFrame from compute_survey()
    surface_lat/lon : float — surface location in geographic degrees
    target_lat/lon  : float — target location in geographic degrees
    surface_n/e : float — surface projected coordinates
    crs_working : str/int — working CRS (e.g. 'EPSG:32630')
    units       : 'metric' or 'imperial'
    save_path   : str — file path to save the HTML map

    Returns
    -------
    folium.Map
    """
    transformer = Transformer.from_crs(crs_working, 'EPSG:4326', always_xy=True)

    # Convert survey stations to lat/lon
    latlons = []
    for _, row in survey_df.iterrows():
        abs_e = surface_e + row['E']
        abs_n = surface_n + row['N']
        lon, lat = transformer.transform(abs_e, abs_n)
        latlons.append((lat, lon))

    centre_lat = (surface_lat + target_lat) / 2.0
    centre_lon = (surface_lon + target_lon) / 2.0

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13)

    # Esri satellite layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
              'World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    # Wellpath polyline
    folium.PolyLine(
        locations=latlons,
        color='blue', weight=3, opacity=0.85,
        tooltip='Wellpath'
    ).add_to(m)

    # Surface marker
    folium.Marker(
        location=[surface_lat, surface_lon],
        popup=f'Surface<br>Lat: {surface_lat:.6f}<br>Lon: {surface_lon:.6f}',
        tooltip='Surface',
        icon=folium.Icon(color='green', icon='home', prefix='fa')
    ).add_to(m)

    # Target marker
    depth_u = DEPTH_UNIT[units]
    folium.Marker(
        location=[target_lat, target_lon],
        popup=(f'Target<br>Lat: {target_lat:.6f}<br>Lon: {target_lon:.6f}'
               f'<br>TVD: {survey_df["TVD"].iloc[-1]:.1f} {depth_u}'),
        tooltip='Target',
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)

    # KOP circle at surface coords (vertical projection) with depth label
    folium.CircleMarker(
        location=[surface_lat, surface_lon],
        radius=8, color='black', fill=True, fill_opacity=0.8,
        popup=f'KOP (depth {survey_df.iloc[1]["TVD"]:.1f} {depth_u})',
        tooltip='KOP'
    ).add_to(m)

    folium.LayerControl().add_to(m)

    m.save(save_path)
    return m


def _plot_map(survey_df, surface_lat, surface_lon, target_lat, target_lon,
              surface_n, surface_e, crs_working, units, save_path, kop_tvd):
    """
    Internal helper that avoids self-import in plot_map.
    Same as plot_map but accepts kop_tvd directly.
    """
    transformer = Transformer.from_crs(crs_working, 'EPSG:4326', always_xy=True)

    latlons = []
    for _, row in survey_df.iterrows():
        abs_e = surface_e + row['E']
        abs_n = surface_n + row['N']
        lon, lat = transformer.transform(abs_e, abs_n)
        latlons.append((lat, lon))

    centre_lat = (surface_lat + target_lat) / 2.0
    centre_lon = (surface_lon + target_lon) / 2.0

    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=13)

    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
              'World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    ).add_to(m)

    folium.PolyLine(
        locations=latlons,
        color='blue', weight=3, opacity=0.85,
        tooltip='Wellpath'
    ).add_to(m)

    depth_u = DEPTH_UNIT[units]

    folium.Marker(
        location=[surface_lat, surface_lon],
        popup=f'Surface<br>Lat: {surface_lat:.6f}<br>Lon: {surface_lon:.6f}',
        tooltip='Surface',
        icon=folium.Icon(color='green', icon='home', prefix='fa')
    ).add_to(m)

    folium.Marker(
        location=[target_lat, target_lon],
        popup=(f'Target<br>Lat: {target_lat:.6f}<br>Lon: {target_lon:.6f}'
               f'<br>TVD: {survey_df["TVD"].iloc[-1]:.1f} {depth_u}'),
        tooltip='Target',
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)

    # KOP circle at surface coords with depth label
    folium.CircleMarker(
        location=[surface_lat, surface_lon],
        radius=8, color='black', fill=True, fill_opacity=0.8,
        popup=f'KOP  TVD: {kop_tvd:.1f} {depth_u}',
        tooltip='KOP'
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(save_path)
    return m


# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------

def export_csv(survey_df, path, units):
    """Export survey DataFrame to CSV."""
    depth_u = DEPTH_UNIT[units]
    dls_u   = DLS_UNIT[units]

    df = survey_df.copy()
    df.columns = [
        f'MD ({depth_u})', f'Inc (deg)', f'Azi (deg)',
        f'TVD ({depth_u})', f'N ({depth_u})', f'E ({depth_u})',
        f'DLS ({dls_u})', f'Dogleg (deg)', f'VS ({depth_u})', 'Closure_Azi (deg)'
    ]
    df.to_csv(path, index=False, float_format='%.4f')
    print(f"  Survey exported → {path}")


def export_excel(survey_df, traj, path, units):
    """Export survey and design summary to Excel (two sheets)."""
    depth_u = DEPTH_UNIT[units]
    dls_u   = DLS_UNIT[units]

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Sheet 1 — Survey
        df = survey_df.copy()
        df.columns = [
            f'MD ({depth_u})', 'Inc (deg)', 'Azi (deg)',
            f'TVD ({depth_u})', f'N ({depth_u})', f'E ({depth_u})',
            f'DLS ({dls_u})', 'Dogleg (deg)', f'VS ({depth_u})', 'Closure_Azi (deg)'
        ]
        df.to_excel(writer, sheet_name='Survey', index=False)

        # Sheet 2 — Design Summary
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
                f"{traj['build_rate']:.2f}", f"{traj['inc_max']:.2f}",
                f"{traj['md_eob']:.2f}", f"{traj['md_eoh']:.2f}",
                f"{traj['md_eoh']:.2f}", f"{traj['H_total']:.2f}",
                f"{traj['target_tvd']:.2f}"
            ]
        }
        pd.DataFrame(summary).to_excel(writer, sheet_name='Design Summary', index=False)

    print(f"  Excel exported  → {path}")


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_well_plan(config):
    """
    High-level entry point.  Accepts a config dict, runs the full workflow,
    prints a summary, generates plots, saves map and exports.

    Parameters
    ----------
    config : dict — see EXAMPLE_CONFIG below

    Returns
    -------
    (survey_df, traj, fig, map_obj)
    """
    import os

    well_cfg  = config['well']
    loc_cfg   = config['location']
    traj_cfg  = config['trajectory']
    out_cfg   = config.get('output', {})

    well_name = well_cfg.get('name', 'Well-1')
    units     = traj_cfg.get('units', 'metric')
    dls_int   = DLS_INTERVAL[units]
    depth_u   = DEPTH_UNIT[units]
    dls_u     = DLS_UNIT[units]

    crs_working = loc_cfg.get('crs', 'EPSG:32630')

    # --- Resolve surface coordinates ---
    if 'lat' in loc_cfg['surface'] and 'lon' in loc_cfg['surface']:
        surface_lat = loc_cfg['surface']['lat']
        surface_lon = loc_cfg['surface']['lon']
        surface_n, surface_e = latlon_to_ne(surface_lat, surface_lon, crs_working)
    else:
        surface_n   = loc_cfg['surface']['northing']
        surface_e   = loc_cfg['surface']['easting']
        surface_lat, surface_lon = ne_to_latlon(surface_n, surface_e, crs_working)

    # --- Resolve target coordinates ---
    if 'lat' in loc_cfg['target'] and 'lon' in loc_cfg['target']:
        target_lat = loc_cfg['target']['lat']
        target_lon = loc_cfg['target']['lon']
        target_n, target_e = latlon_to_ne(target_lat, target_lon, crs_working)
    else:
        target_n   = loc_cfg['target']['northing']
        target_e   = loc_cfg['target']['easting']
        target_lat, target_lon = ne_to_latlon(target_n, target_e, crs_working)

    target_tvd = loc_cfg['target']['tvd']

    # --- Design trajectory ---
    traj = design_trajectory(
        surface_n, surface_e, target_n, target_e, target_tvd,
        kop_tvd       = traj_cfg['kop_tvd'],
        build_rate    = traj_cfg['build_rate'],
        surface_dls   = traj_cfg.get('surface_dls', 0.0),
        units         = units,
        survey_interval = traj_cfg.get('survey_interval', dls_int)
    )

    survey_interval = traj_cfg.get('survey_interval', dls_int)

    # --- Generate stations and compute survey ---
    stations   = generate_stations(traj, survey_interval, units)
    survey_df  = compute_survey(stations, dls_int)

    # --- Survey tool uncertainty ---
    survey_cfg    = config.get('survey', {})
    tool_sections = survey_cfg.get('tool_sections', [
        {'from_md': 0, 'to_md': None, 'code': 'MWD'}
    ])
    confidence    = survey_cfg.get('confidence', 2.0)
    uncertainty_df = compute_uncertainty(survey_df, tool_sections, units, confidence)

    # --- Print summary ---
    lat_dir = 'N' if surface_lat >= 0 else 'S'
    lon_dir = 'E' if surface_lon >= 0 else 'W'
    t_lat_dir = 'N' if target_lat >= 0 else 'S'
    t_lon_dir = 'E' if target_lon >= 0 else 'W'

    sep  = '=' * 60
    hsep = '─' * 58  # ──────

    print(sep)
    print(f"  WELL DESIGN SUMMARY — {well_name}    [{units}]")
    print(sep)
    print(f"  Surface Location:  {abs(surface_lat):.6f}°{lat_dir}  "
          f"{abs(surface_lon):.6f}°{lon_dir}")
    print(f"  Surface N/E:       {surface_n:.1f} {depth_u} N,  "
          f"{surface_e:.1f} {depth_u} E  ({crs_working})")
    print(f"  Target Location:   {abs(target_lat):.6f}°{t_lat_dir}  "
          f"{abs(target_lon):.6f}°{t_lon_dir}")
    print(f"  Target N/E:        {target_n:.1f} {depth_u} N,  "
          f"{target_e:.1f} {depth_u} E")
    print(f"  Target TVD:        {target_tvd:.1f} {depth_u}")
    print(f"  Horizontal Dep:    {traj['H_total']:.1f} {depth_u}"
          f"   Azimuth: {traj['azimuth']:06.2f}°")
    print(f"  {hsep}")
    if traj['surface_dls'] > 1e-10:
        print(f"  Surface DLS:       {traj['surface_dls']:.2f} {dls_u}"
              f"  →  Inc at KOP: {traj['inc_at_kop']:.1f}°")
    print(f"  KOP:               {traj['kop_tvd']:.1f} {depth_u} TVD"
          f"  /  {traj['md_kop']:.1f} {depth_u} MD")
    print(f"  Build Rate:        {traj['build_rate']:.2f} {dls_u}")
    print(f"  Max Inclination:   {traj['inc_max']:.2f}°")
    print(f"  EOB:               {traj['md_eob']:.1f} {depth_u} MD")
    print(f"  Hold Section:      {traj['md_eoh']:.1f} {depth_u} MD")
    print(f"  TD (MD):           {traj['md_eoh']:.1f} {depth_u}")
    # Survey tool summary
    print(f'  {hsep}')
    print('  Survey Tool Sections:')
    for sec in tool_sections:
        to   = f"{sec['to_md']:.0f} {depth_u}" if sec.get('to_md') else 'TD'
        frm  = f"{sec['from_md']:.0f} {depth_u}"
        tool = SURVEY_TOOLS.get(sec['code'], {})
        print(f"    {frm} → {to}:  {sec['code']} — {tool.get('name', '')}")
    td_unc = uncertainty_df.iloc[-1]
    print(f'  Position Uncertainty at TD (~95%):')
    print(f"    Lateral: ±{td_unc['σ_lat']:.1f} {depth_u}   "
          f"High-side: ±{td_unc['σ_hs']:.1f} {depth_u}   "
          f"TVD: ±{td_unc['σ_tvd']:.1f} {depth_u}")
    print(sep)

    # --- Plots ---
    fig = plot_plan_and_section(
        survey_df, traj, surface_n, surface_e,
        target_n, target_e, units, well_name=well_name,
        uncertainty_df=uncertainty_df
    )

    fig3d = plot_3d(survey_df, traj, units, well_name=well_name)

    # --- Map ---
    out_dir  = out_cfg.get('directory', '.')
    map_path = os.path.join(out_dir, f"{well_name}_map.html")

    map_obj = _plot_map(
        survey_df, surface_lat, surface_lon, target_lat, target_lon,
        surface_n, surface_e, crs_working, units, map_path,
        kop_tvd=traj['kop_tvd']
    )
    print(f"  Map saved        → {map_path}")

    # --- Export ---
    csv_path   = os.path.join(out_dir, f"{well_name}_survey.csv")
    excel_path = os.path.join(out_dir, f"{well_name}_survey.xlsx")
    export_csv(survey_df, csv_path, units)
    export_excel(survey_df, traj, excel_path, units)

    plt.show()

    return survey_df, traj, uncertainty_df, fig, map_obj


# ---------------------------------------------------------------------------
# EXAMPLE CONFIG — North Sea well
# ---------------------------------------------------------------------------

EXAMPLE_CONFIG = {
    'well': {
        'name': 'Well-1A',
    },
    'location': {
        'crs': 'EPSG:32630',          # WGS84 / UTM zone 30N
        'surface': {
            'lat': 57.600000,
            'lon':  1.850000,
        },
        'target': {
            'lat': 57.608000,
            'lon':  1.862000,
            'tvd': 2800.0,            # metres
        },
    },
    'trajectory': {
        'kop_tvd':         600.0,     # m
        'build_rate':        3.5,     # deg/30m
        'surface_dls':       0.5,     # deg/30m  (0 = vertical)
        'units':          'metric',
        'survey_interval':  30.0,     # m
    },
    'output': {
        'directory': '.',
    },
    'survey': {
        # tool_sections: list of {from_md, to_md (None=TD), code}
        # Available codes: BLIND, INC_ONLY, SINGLE_SHOT, MULTI_SHOT,
        #                  MWD, MWD_IFR, MWD_IFR_MS, GYRO_SS, GYRO_CONT
        'tool_sections': [
            {'from_md':   0.0, 'to_md': 600.0, 'code': 'INC_ONLY'},    # surface casing
            {'from_md': 600.0, 'to_md':   None, 'code': 'MWD_IFR_MS'}, # directional section
        ],
        'confidence': 2.0,   # 1.0=1σ(68%), 2.0≈95%, 2.576≈99%
    },
}


if __name__ == '__main__':
    survey_df, traj, unc_df, fig, map_obj = run_well_plan(EXAMPLE_CONFIG)
