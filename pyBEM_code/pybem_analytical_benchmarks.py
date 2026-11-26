import csv
from pathlib import Path
import numpy as np

# ==============================================================================
# STANDALONE BENCHMARK SWITCHES & PATHS (Edit here before running)
# ==============================================================================
C_MM_S = 343041.0            # Speed of sound in mm/s (e.g. sqrt(0.1424 / 1.21e-12) or 343000.0)
L_PIPE_MM = 1000.0           # Default straight pipe length in mm
F_MAX_HZ = 1200.0            # Upper frequency limit for modal / peak / trough checks

RUN_1M_PIPE_SUITE = True
PIPE_1M_DIR = Path(r"F:\FEA\PRJs\acoustics\pipes\1m_pipes_suite\1Hz")
PIPE_1M_SURFACE = None       # None = auto-pick per case's preferred_surfaces ("inlet"/"outlet"), or force e.g. "inlet"
PIPE_1M_METRIC = "swl_mag_db"  # Energy-based SWL metric by default (e.g. "swl_mag_db", "swl_real_db", "tsw_mag_mw")
PIPE_1M_MIN_ABS_PROM = 3.0   # Minimum absolute prominence in metric units (e.g. 3.0 dB for SWL; None to disable)
PIPE_1M_MATCH_TOL_HZ = 35.0  # Max |f_sim - f_ref| (Hz) to pair an extremum with an analytical mode
PIPE_1M_PATTERNS = {
    "CASE 1": "01*power*.csv",
    "CASE 2": "02*power*.csv",
    "CASE 3": "03*power*.csv",
    "CASE 4": "04*power*.csv",
    "CASE 5": "05*power*.csv",
    "CASE 6": "06*power*.csv",
}

RUN_IMPEDANCE_DUCT_CHECK = False
DUCT_CSV = Path(r"path\to\duct_power.csv")
DUCT_SURFACE = "outlet"
DUCT_PARAMS = {"L": 1000.0, "c": C_MM_S, "z_load": 2.0 + 0.5j}

RUN_DEMO_SMOKE_TEST = False
# ==============================================================================

DEFAULT_C_MM_S = C_MM_S
DEFAULT_L_PIPE_MM = L_PIPE_MM

METRIC_UNITS: dict[str, str] = {
    "swl_real_dba": "dBA",
    "swl_real_db": "dB",
    "swl_imag_dba": "dBA",
    "swl_imag_db": "dB",
    "swl_mag_dba": "dBA",
    "swl_mag_db": "dB",
    "w_imag_over_real": "ratio",
    "tsw_real_mw": "mW",
    "tsw_imag_mw": "mW",
    "tsw_mag_mw": "mW",
    "r_mag": "ratio",
    "r_real": "ratio",
    "r_imag": "ratio",
    "alpha": "ratio",
    "spl_dba": "dBA",
    "spl_db": "dB",
    "svl_db": "dB",
    "p_rms_mpa": "MPa",
    "v_mean_mm_s": "mm/s",
    "v_mag_mm_s": "mm/s",
}


def _metric_unit_label(metric_key: str) -> str:
    return METRIC_UNITS.get(metric_key, "units")


def _normalize_header_str(col: str) -> str:
    parts = [p.strip().lower() for p in col.split("|")]
    return " | ".join(parts)


def _classify_metric_suffix(norm_col: str) -> str | None:
    rules = [
        ("swl_real_dba", "| swl_real (dba)"),
        ("swl_real_db", "| swl_real (db)"),
        ("swl_imag_dba", "| swl_imag (dba)"),
        ("swl_imag_db", "| swl_imag (db)"),
        ("swl_mag_dba", "| swl_mag (dba)"),
        ("swl_mag_db", "| swl_mag (db)"),
        ("w_imag_over_real", "| w_imag/w_real"),
        ("tsw_real_mw", "| tsw_real (mw)"),
        ("tsw_imag_mw", "| tsw_imag (mw)"),
        ("tsw_mag_mw", "| tsw_mag (mw)"),
        ("r_mag", "| r_mag"),
        ("r_real", "| r_real"),
        ("r_imag", "| r_imag"),
        ("alpha", "| alpha"),
        ("spl_dba", "| spl (dba)"),
        ("spl_db", "| spl (db)"),
        ("svl_db", "| svl (db)"),
        ("p_rms_mpa", "| p_rms (mpa)"),
        ("v_mean_mm_s", "| v_mean (mm/s)"),
        ("v_mag_mm_s", "| v_mag (mm/s)"),
    ]
    for canonical, token in rules:
        if token in norm_col:
            return canonical
    return None


def _resolve_freq_header(headers: list[str], freq_candidates: list[str] | None = None) -> str:
    candidates = freq_candidates or ["Freq(Hz)", "freq", "frequency", "freq_hz", "f_hz", "f"]
    lower_map = {col.strip().lower(): col for col in headers if col is not None}
    for cand in candidates:
        key = cand.strip().lower()
        if key in lower_map:
            return lower_map[key]
    raise KeyError(
        f"Could not find frequency column. Tried {candidates}. Available columns: {headers}"
    )


def load_pybem_csv(csv_path: str | Path,
                   freq_col: list[str] | None = None) -> dict:
    """
    Load a pyBEM power-calculator CSV using stdlib csv + numpy.
    Automatically parses `hdr_base | ... | <metric>` headers per surface.
    """
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV file '{path}' has no header row.")
        headers = [h for h in reader.fieldnames if h is not None]
        rows = list(reader)

    f_col = _resolve_freq_header(headers, freq_col)

    col_map: dict[str, tuple[str, str]] = {}
    for col in headers:
        if col == f_col:
            continue
        norm_col = _normalize_header_str(col)
        metric_key = _classify_metric_suffix(norm_col)
        if metric_key is None:
            continue
        raw_parts = [p.strip() for p in col.split("|")]
        surface_name = raw_parts[0] if raw_parts and raw_parts[0] else "default"
        col_map[col] = (surface_name, metric_key)

    def _to_float_array(col_name: str) -> np.ndarray:
        vals = []
        for row in rows:
            raw = row.get(col_name, "")
            s = raw.strip() if raw is not None else ""
            if s == "" or s.lower() in ("nan", "null", "none"):
                vals.append(np.nan)
                continue
            try:
                vals.append(float(s))
            except ValueError:
                cleaned = s.strip("()")
                vals.append(float(abs(complex(cleaned))))
        return np.asarray(vals, dtype=float)

    surfaces: dict[str, dict[str, np.ndarray]] = {}
    for col_name, (surface_name, metric_key) in col_map.items():
        surf_dict = surfaces.setdefault(surface_name, {})
        if metric_key not in surf_dict:
            surf_dict[metric_key] = _to_float_array(col_name)

    return {
        "freq": _to_float_array(f_col),
        "surfaces": surfaces,
    }


def _curve_swing(arr: np.ndarray, mask: np.ndarray | None = None) -> float:
    sub = arr[mask] if mask is not None else arr
    finite = sub[np.isfinite(sub)]
    if len(finite) < 2:
        return 0.0
    return float(np.max(finite) - np.min(finite))


def select_surface_metrics(sim: dict,
                           metric_key: str,
                           surface_name: str | None = None,
                           preferred_surfaces: tuple[str, ...] | list[str] | None = None,
                           band_mask: np.ndarray | None = None,
                           min_swing_tol: float = 1e-4) -> tuple[str, np.ndarray]:
    """
    Extract a metric array from `sim["surfaces"]`:
      1. If `surface_name` is explicitly given, use it (case-insensitive).
      2. Else evaluate candidate surfaces in `preferred_surfaces` order (then remaining
         surfaces with `metric_key`), preferring the first one with non-flat variation
         (`swing >= min_swing_tol`), and falling back to the first matching surface
         if all matching curves are flat across frequency.
    """
    surfaces: dict[str, dict[str, np.ndarray]] = sim.get("surfaces", {})
    if not surfaces:
        raise KeyError("No surface metric columns found in CSV.")

    lower_to_actual = {k.strip().lower(): k for k in surfaces}

    if surface_name is not None:
        key = surface_name.strip().lower()
        if key not in lower_to_actual:
            raise KeyError(
                f"Surface '{surface_name}' not found. Available surfaces: {list(surfaces.keys())}"
            )
        actual_surf = lower_to_actual[key]
        if metric_key not in surfaces[actual_surf]:
            raise KeyError(
                f"Metric '{metric_key}' not found on surface '{actual_surf}'. "
                f"Available metrics: {list(surfaces[actual_surf].keys())}"
            )
        return actual_surf, surfaces[actual_surf][metric_key]

    ordered_candidates: list[str] = []
    if preferred_surfaces:
        for pref in preferred_surfaces:
            actual = lower_to_actual.get(pref.strip().lower())
            if actual and actual not in ordered_candidates:
                ordered_candidates.append(actual)
    for actual in surfaces:
        if actual not in ordered_candidates:
            ordered_candidates.append(actual)

    first_valid: tuple[str, np.ndarray] | None = None
    for actual_surf in ordered_candidates:
        metrics = surfaces[actual_surf]
        if metric_key not in metrics:
            continue
        arr = metrics[metric_key]
        if first_valid is None:
            first_valid = (actual_surf, arr)
        if _curve_swing(arr, band_mask) >= min_swing_tol:
            return actual_surf, arr

    if first_valid is not None:
        return first_valid

    available_all = {s: list(m.keys()) for s, m in surfaces.items()}
    raise KeyError(f"Metric '{metric_key}' not found on any surface. Available: {available_all}")


def straight_pipe_modes(L: float = DEFAULT_L_PIPE_MM,
                        c: float = DEFAULT_C_MM_S,
                        bc: str = "closed-closed",
                        n_max: int = 6,
                        delta_L: float = 0.0) -> np.ndarray:
    """
    Analytical 1D standing-wave resonant frequencies (Hz) for a straight pipe.
    L and delta_L in mm; c in mm/s.
    """
    L_eff = L + delta_L
    if L_eff <= 0:
        raise ValueError("Effective length L_eff must be positive.")

    bc_norm = bc.strip().lower()
    ns = np.arange(1, n_max + 1, dtype=float)
    if bc_norm in ("closed-closed", "open-open", "rigid-rigid", "open-open-ideal"):
        return ns * c / (2.0 * L_eff)
    elif bc_norm in ("closed-open", "open-closed", "rigid-open", "closed-pressure-release"):
        return (2.0 * ns - 1.0) * c / (4.0 * L_eff)
    else:
        raise ValueError(f"Unsupported boundary condition '{bc}'.")


def impedance_terminated_duct(freqs: np.ndarray,
                              L: float = DEFAULT_L_PIPE_MM,
                              c: float = DEFAULT_C_MM_S,
                              z_load: complex = np.inf + 0j) -> dict:
    """
    1D plane-wave input normalized impedance z_in, reflection coefficient R,
    absorption alpha = 1 - |R|^2, and reactive/active power ratio at x = 0
    (distance L in mm upstream of load at x = L; c in mm/s).
    """
    freqs = np.asarray(freqs, dtype=float)
    omega = 2.0 * np.pi * freqs
    k = omega / c
    kl = k * L

    if np.isinf(z_load.real) and np.isinf(z_load.imag):
        z_in = -1j / np.tan(kl)
    else:
        t = np.tan(kl)
        z_in = (z_load + 1j * t) / (1.0 + 1j * z_load * t)

    r_complex = (z_in - 1.0) / (z_in + 1.0)
    r_mag = np.abs(r_complex)
    alpha = 1.0 - r_mag ** 2

    with np.errstate(divide="ignore", invalid="ignore"):
        w_imag_over_real = np.where(np.abs(z_in.real) > 1e-15, z_in.imag / z_in.real, np.nan)

    return {
        "freq": freqs,
        "z_in": z_in,
        "r_complex": r_complex,
        "r_mag": r_mag,
        "alpha": alpha,
        "w_imag_over_real": w_imag_over_real,
    }


def rigid_box_modes(Lx: float, Ly: float, Lz: float,
                    c: float = DEFAULT_C_MM_S,
                    n_modes: int = 12, max_idx: int = 5) -> list[dict]:
    """
    Analytical standing-wave eigenfrequencies (Hz) for a closed rigid rectangular cavity.
    Dimensions Lx, Ly, Lz in mm; c in mm/s.
    """
    modes = []
    for l in range(max_idx + 1):
        for m in range(max_idx + 1):
            for n in range(max_idx + 1):
                if l == 0 and m == 0 and n == 0:
                    continue
                freq = 0.5 * c * np.sqrt(
                    (l / Lx) ** 2 + (m / Ly) ** 2 + (n / Lz) ** 2
                )
                modes.append({"freq": freq, "mode": (l, m, n)})

    modes.sort(key=lambda item: item["freq"])
    return modes[:n_modes]


def expansion_chamber_tl(freqs: np.ndarray,
                         L_chamber: float,
                         c: float = DEFAULT_C_MM_S,
                         area_chamber: float = 1.0,
                         area_pipe: float = 1.0) -> np.ndarray:
    """
    1D plane-wave Transmission Loss (dB) for a simple expansion chamber
    of length L_chamber (mm) and area S_c (mm^2) connected to inlet/outlet pipes
    of area S_i (mm^2), with sound speed c in mm/s.
    """
    freqs = np.asarray(freqs, dtype=float)
    k = 2.0 * np.pi * freqs / c
    m = area_chamber / area_pipe
    term = 0.25 * (m - 1.0 / m) ** 2 * (np.sin(k * L_chamber) ** 2)
    return 10.0 * np.log10(1.0 + term)


def helmholtz_freq_hz(V: float, S: float, L_eff: float,
                      c: float = DEFAULT_C_MM_S) -> float:
    """
    1D lumped Helmholtz resonance frequency (Hz).
    Keep V in mm^3, S in mm^2, L_eff in mm, and c in mm/s.
    """
    return (c / (2.0 * np.pi)) * np.sqrt(S / (V * L_eff))


PIPE_1M_CASES = [
    {
        "case": "CASE 1",
        "bc_desc": "PRES @ INLET & rigid OUTLET",
        "regime": "quarter-wave",
        "mode_bc": "closed-open",
        "preferred_surfaces": ("inlet", "outlet"),
        "feature": "peaks",
        "expect_modes": True,
    },
    {
        "case": "CASE 2",
        "bc_desc": "PRES @ INLET & OUTLET (open both ends)",
        "regime": "open-open (no resonant peaks expected)",
        "mode_bc": None,
        "preferred_surfaces": ("outlet", "inlet"),
        "feature": "peaks",
        "expect_modes": False,
    },
    {
        "case": "CASE 3",
        "bc_desc": "PRES @ INLET & IMPE(z = rho*c) @ OUTLET",
        "regime": "anechoic termination (no resonant peaks expected)",
        "mode_bc": None,
        "preferred_surfaces": ("inlet", "outlet"),
        "feature": "peaks",
        "expect_modes": False,
    },
    {
        "case": "CASE 4",
        "bc_desc": "VELO @ INLET & rigid OUTLET",
        "regime": "half-wave closed-closed",
        "mode_bc": "closed-closed",
        "preferred_surfaces": ("inlet", "outlet"),
        "feature": "peaks",
        "expect_modes": True,
    },
    {
        "case": "CASE 5",
        "bc_desc": "[VELO & IMPE(z = rho*c)] @ INLET & rigid OUTLET",
        "regime": "quarter-wave (notch/dip at modes on inlet)",
        "mode_bc": "closed-open",
        "preferred_surfaces": ("inlet", "outlet"),
        "feature": "troughs",
        "expect_modes": True,
    },
    {
        "case": "CASE 6",
        "bc_desc": "VELO @ INLET & IMPE(z = rho*c) @ OUTLET",
        "regime": "anechoic termination (no resonant peaks expected)",
        "mode_bc": None,
        "preferred_surfaces": ("inlet", "outlet"),
        "feature": "peaks",
        "expect_modes": False,
    },
]


def print_1m_pipe_reference_table(L_mm: float = DEFAULT_L_PIPE_MM,
                                  c_mm_s: float = DEFAULT_C_MM_S,
                                  f_max_hz: float = F_MAX_HZ) -> None:
    print(f"\n=== 1m Straight Pipe Analytical Reference (L={L_mm} mm, c={c_mm_s} mm/s, <= {f_max_hz} Hz) ===")
    for item in PIPE_1M_CASES:
        if item["expect_modes"] and item["mode_bc"]:
            raw_modes = straight_pipe_modes(L=L_mm, c=c_mm_s, bc=item["mode_bc"], n_max=20)
            in_band = raw_modes[raw_modes <= f_max_hz]
            modes_str = ", ".join(f"{f:.2f}" for f in in_band) + " Hz"
        else:
            modes_str = "None expected"
        surf_hint = "/".join(item["preferred_surfaces"])
        print(f"  {item['case']} | {item['bc_desc']:50s} | [{surf_hint:13s}] | {modes_str}")


def _parabolic_refine(f: np.ndarray, y: np.ndarray, idx: int) -> tuple[float, float]:
    """
    Sub-bin parabolic interpolation around interior index `idx`.
    Returns (f_refined, y_refined). Falls back safely to grid sample if curvature is degenerate.
    """
    if idx <= 0 or idx >= len(y) - 1:
        return float(f[idx]), float(y[idx])

    y0, y1, y2 = float(y[idx - 1]), float(y[idx]), float(y[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-18:
        return float(f[idx]), float(y[idx])

    delta = 0.5 * (y0 - y2) / denom
    if abs(delta) > 1.0:
        return float(f[idx]), float(y[idx])

    df_left = float(f[idx] - f[idx - 1])
    df_right = float(f[idx + 1] - f[idx])
    df_step = df_left if delta < 0 else df_right

    f_ref = float(f[idx]) + delta * df_step
    y_ref = y1 - 0.25 * (y0 - y2) * delta
    return f_ref, y_ref


def find_extrema_refined(freqs: np.ndarray,
                         values: np.ndarray,
                         kind: str = "peaks",
                         min_prominence_frac: float = 0.12,
                         min_abs_prom: float | None = None) -> list[dict]:
    """
    Find prominent local peaks (`kind='peaks'`) or troughs (`kind='troughs'`) with
    sub-bin parabolic frequency interpolation and both relative + absolute prominence filters.
    """
    freqs = np.asarray(freqs, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(freqs) & np.isfinite(values)
    f = freqs[mask]
    y_raw = values[mask]
    if len(y_raw) < 3:
        return []

    y = y_raw if kind == "peaks" else -y_raw
    total_span = float(np.max(y) - np.min(y))
    if total_span <= 1e-12:
        return []

    rel_thresh = min_prominence_frac * total_span
    thresh = max(rel_thresh, float(min_abs_prom)) if min_abs_prom is not None else rel_thresh
    found: list[dict] = []

    for i in range(1, len(y) - 1):
        is_local = (y[i] >= y[i - 1] and y[i] >= y[i + 1]) and (y[i] > y[i - 1] or y[i] > y[i + 1])
        if not is_local:
            continue

        left_min = float(np.min(y[:i]))
        right_min = float(np.min(y[i + 1:]))
        prom = min(float(y[i]) - left_min, float(y[i]) - right_min)
        if prom < thresh:
            continue

        f_ref, y_ref_signed = _parabolic_refine(f, y, i)
        found.append({
            "f_grid": float(f[i]),
            "f_interp": f_ref,
            "val_grid": float(y_raw[i]),
            "val_interp": float(y_ref_signed if kind == "peaks" else -y_ref_signed),
            "prominence": prom,
            "prom_pct_span": (prom / total_span) * 100.0,
        })

    return found


def compare_extrema(extrema: list[dict],
                    ref_freqs: np.ndarray,
                    label: str = "Extrema",
                    unit_label: str = "dB",
                    feat_kind: str = "peaks",
                    max_match_df_hz: float | None = 35.0) -> None:
    """
    Compare sub-bin interpolated frequencies against analytical reference frequencies.
    """
    prom_tag = "peak_prom" if feat_kind == "peaks" else "trough_prom"
    print(f"\n--- {label} Comparison (Parabolic Sub-Bin) ---")
    if len(extrema) == 0 or len(ref_freqs) == 0:
        f_list = [round(e["f_interp"], 2) for e in extrema]
        print(f"  pyBEM (Hz): {f_list}")
        print(f"  Ref   (Hz): {np.round(ref_freqs, 2)}")
        return

    unmatched_hz: list[float] = []
    for item in extrema:
        sp = item["f_interp"]
        idx = int(np.argmin(np.abs(ref_freqs - sp)))
        rp = float(ref_freqs[idx])
        df_hz = sp - rp
        if max_match_df_hz is not None and abs(df_hz) > max_match_df_hz:
            unmatched_hz.append(sp)
            continue
        pct = (df_hz / rp) * 100.0 if rp != 0 else np.nan
        print(
            f"  pyBEM: {sp:7.2f} Hz (grid {item['f_grid']:6.1f}) | "
            f"Ref: {rp:7.2f} Hz | dF: {df_hz:+7.2f} Hz ({pct:+6.2f}%) | "
            f"{prom_tag}={item['prominence']:.2f} {unit_label} ({item['prom_pct_span']:.0f}% of max_dyn_range)"
        )

    if unmatched_hz:
        print(f"  Unmatched extrema (> {max_match_df_hz:.1f} Hz from nearest mode): {[round(x, 2) for x in unmatched_hz]}")


def run_1m_pipe_suite(suite_dir: str | Path,
                      file_patterns: dict[str, str],
                      surface_name: str | None = None,
                      metric_key: str = "swl_mag_db",
                      L_mm: float = DEFAULT_L_PIPE_MM,
                      c_mm_s: float = DEFAULT_C_MM_S,
                      f_max_hz: float = F_MAX_HZ,
                      min_abs_prom: float | None = PIPE_1M_MIN_ABS_PROM,
                      max_match_df_hz: float | None = PIPE_1M_MATCH_TOL_HZ) -> None:
    folder = Path(suite_dir)
    unit_label = _metric_unit_label(metric_key)
    print_1m_pipe_reference_table(L_mm=L_mm, c_mm_s=c_mm_s, f_max_hz=f_max_hz)

    if not folder.is_dir():
        print(f"\n[!] 1m_pipe folder not found: {folder}")
        return

    for item in PIPE_1M_CASES:
        cid = item["case"]
        pattern = file_patterns.get(cid)
        if not pattern:
            continue

        matches = sorted(folder.glob(pattern))
        if not matches:
            print(f"\n[{cid}] No CSV matched '{pattern}' in {folder}")
            continue

        for csv_file in matches:
            sim = load_pybem_csv(csv_file)
            freqs = sim["freq"]
            in_band = freqs <= f_max_hz
            f_sub = freqs[in_band]

            try:
                used_surf, y_vals = select_surface_metrics(
                    sim,
                    metric_key=metric_key,
                    surface_name=surface_name,
                    preferred_surfaces=item.get("preferred_surfaces"),
                    band_mask=in_band,
                )
            except KeyError as exc:
                print(f"\n--- {cid}: {csv_file.name} ---\n  [!] {exc}")
                continue

            y_sub = y_vals[in_band]
            dyn_range = _curve_swing(y_sub)
            feat_kind = item.get("feature", "peaks")

            print(
                f"\n--- {cid}: {csv_file.name} | Surface='{used_surf}' | "
                f"Metric='{metric_key}' | max_dyn_range={dyn_range:.2f} {unit_label} ---"
            )
            print(f"  BCs: {item['bc_desc']}")

            if not item["expect_modes"]:
                ext = find_extrema_refined(
                    f_sub,
                    y_sub,
                    kind=feat_kind,
                    min_prominence_frac=0.20,
                    min_abs_prom=min_abs_prom,
                )
                f_det = [round(e["f_interp"], 2) for e in ext]
                print(f"  Expected: No resonant modes. Prominent {feat_kind}: {f_det}")
                continue

            ref_modes = straight_pipe_modes(L=L_mm, c=c_mm_s, bc=item["mode_bc"], n_max=20)
            ref_in_band = ref_modes[ref_modes <= f_max_hz]
            print(f"  Analytical modes <= {f_max_hz} Hz: {np.round(ref_in_band, 2)}")

            ext = find_extrema_refined(
                f_sub,
                y_sub,
                kind=feat_kind,
                min_prominence_frac=0.10,
                min_abs_prom=min_abs_prom,
            )
            compare_extrema(
                ext,
                ref_in_band,
                f"{cid} ({used_surf} {metric_key} {feat_kind})",
                unit_label=unit_label,
                feat_kind=feat_kind,
                max_match_df_hz=max_match_df_hz,
            )


def curve_error_summary(sim_freq: np.ndarray, sim_val: np.ndarray,
                        ana_freq: np.ndarray, ana_val: np.ndarray,
                        metric_name: str) -> dict:
    sim_freq = np.asarray(sim_freq, dtype=float)
    sim_val = np.asarray(sim_val, dtype=float)
    ana_freq = np.asarray(ana_freq, dtype=float)
    ana_val = np.asarray(ana_val, dtype=float)

    valid = np.isfinite(sim_freq) & np.isfinite(sim_val)
    f_s = sim_freq[valid]
    y_s = sim_val[valid]

    in_range = (f_s >= np.min(ana_freq)) & (f_s <= np.max(ana_freq))
    f_eval = f_s[in_range]
    y_sim = y_s[in_range]
    y_ana = np.interp(f_eval, ana_freq, ana_val)

    finite_pair = np.isfinite(y_sim) & np.isfinite(y_ana)
    diff = y_sim[finite_pair] - y_ana[finite_pair]
    abs_err = np.abs(diff)

    return {
        "metric": metric_name,
        "n_pts": int(np.sum(finite_pair)),
        "mae": float(np.mean(abs_err)) if len(abs_err) else np.nan,
        "rmse": float(np.sqrt(np.mean(diff ** 2))) if len(diff) else np.nan,
        "max_abs_err": float(np.max(abs_err)) if len(abs_err) else np.nan,
        "mean_bias": float(np.mean(diff)) if len(diff) else np.nan,
    }


def compare_impedance_duct_csv(csv_path: str | Path,
                               params: dict,
                               surface_name: str | None = None,
                               metrics_to_compare: list[str] | None = None) -> None:
    sim = load_pybem_csv(csv_path)
    f_sim = sim["freq"]
    ana = impedance_terminated_duct(f_sim, **params)
    requested = metrics_to_compare or ["r_mag", "alpha", "w_imag_over_real"]

    print("\n==========================================")
    print(f"Benchmark Case : impedance_duct ({Path(csv_path).name})")
    print("==========================================")

    for m in requested:
        if m not in ana:
            continue
        try:
            used_surf, sim_arr = select_surface_metrics(sim, metric_key=m, surface_name=surface_name)
        except KeyError:
            continue
        err = curve_error_summary(f_sim, sim_arr, ana["freq"], ana[m], f"{used_surf}:{m}")
        print(
            f"  [{err['metric']}] n={err['n_pts']} | "
            f"MAE={err['mae']:.5f} | RMSE={err['rmse']:.5f} | "
            f"MaxErr={err['max_abs_err']:.5f} | Bias={err['mean_bias']:+.5f}"
        )


if __name__ == "__main__":
    if RUN_1M_PIPE_SUITE:
        run_1m_pipe_suite(
            suite_dir=PIPE_1M_DIR,
            file_patterns=PIPE_1M_PATTERNS,
            surface_name=PIPE_1M_SURFACE,
            metric_key=PIPE_1M_METRIC,
            L_mm=L_PIPE_MM,
            c_mm_s=C_MM_S,
            f_max_hz=F_MAX_HZ,
            min_abs_prom=PIPE_1M_MIN_ABS_PROM,
            max_match_df_hz=PIPE_1M_MATCH_TOL_HZ,
        )

    if RUN_IMPEDANCE_DUCT_CHECK:
        compare_impedance_duct_csv(
            csv_path=DUCT_CSV,
            params=DUCT_PARAMS,
            surface_name=DUCT_SURFACE,
        )

    if RUN_DEMO_SMOKE_TEST:
        demo_csv = Path("demo_pybem_duct.csv")
        f_grid = np.linspace(50.0, 500.0, 451)
        ref = impedance_terminated_duct(f_grid, L=L_PIPE_MM, c=C_MM_S, z_load=2.0 + 0.5j)

        rng = np.random.default_rng(42)
        sim_r_mag = np.clip(ref["r_mag"] + rng.normal(0.0, 0.008, size=len(f_grid)), 0.0, 1.0)
        sim_alpha = 1.0 - sim_r_mag ** 2
        sim_swl = 80.0 + 10.0 * np.log10(np.maximum(1.0 - sim_alpha, 1e-6))

        header = (
            "Freq(Hz),"
            "INLET | TSW_mag (mW) = 1.25 | SWL_mag (dB),"
            "OUTLET | TSW_mag (mW) = 0.42 | SWL_mag (dB),"
            "OUTLET | W_imag/W_real,"
            "OUTLET | R_mag,"
            "OUTLET | alpha"
        )
        data = np.column_stack([
            f_grid,
            sim_swl + 3.0,
            sim_swl,
            ref["w_imag_over_real"],
            sim_r_mag,
            sim_alpha,
        ])
        np.savetxt(demo_csv, data, delimiter=",", header=header, comments="")

        compare_impedance_duct_csv(
            csv_path=demo_csv,
            params={"L": L_PIPE_MM, "c": C_MM_S, "z_load": 2.0 + 0.5j},
            surface_name="OUTLET",
        )

        if demo_csv.exists():
            demo_csv.unlink()