"""Prior Experiment Dashboard — interactive exploration of attribution maps.

Panels:
  ┌────────────── Controls ──────────────────────────────────────────────────┐
  │  Dataset ▾  Arch ▾  Mode ▾  Seed ▾  Value ▾  Colormap ▾  Thresh  Top-N │
  │  [⚡ Plotly] / [🧠 nilearn]  toggle                                       │
  ├──── Slice sliders (Plotly mode) ─────────────────────────────────────────┤
  ├──────────────────── Brain viewer (full width) ────────────────────────── ┤
  ├────────────────┬───────────────────────────────────┬─────────────────────┤
  │  Seed agree    │  Top-N region bar (with std_abs)  │  Cross-mode table   │
  └────────────────┴───────────────────────────────────┴─────────────────────┘

Usage:
    python -m brainage_agg.analysis.prior_experiment_dashboard
    python -m brainage_agg.analysis.prior_experiment_dashboard --port 8051 --debug
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import Input, Output, State, dcc, html

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PRIOR_DIR = ROOT / "outputs" / "prior_experiment"
_BRAIN_IFRAME_HEIGHT_DEFAULT = 460
_BRAIN_GRAPH_HEIGHT = 460
_BRAIN_HTML_CACHE_SIZE = 128

_FS_AFFINE = np.array([
    [-1, 0, 0, 128],
    [0,  0, 1, -128],
    [0, -1, 0, 128],
    [0,  0, 0,   1],
], dtype=float)

_ABS_CMAPS    = ["hot", "viridis", "plasma", "YlOrRd"]
_SGN_CMAPS    = ["RdBu_r", "coolwarm", "bwr"]
_SIGDIG_CMAPS = ["RdYlGn", "viridis", "plasma", "YlGn"]

_TEMPLATEFLOW   = Path.home() / ".cache/templateflow/tpl-MNI152NLin2009cAsym"
_DKT31_ATLAS_PATH = _TEMPLATEFLOW / "tpl-MNI152NLin2009cAsym_res-02_desc-DKT31_dseg.nii.gz"
_MNI_T1W_PATH     = _TEMPLATEFLOW / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_T1w.nii.gz"

_BTN_ACTIVE = {
    "fontSize": "12px", "padding": "4px 10px", "cursor": "pointer",
    "background": "#2f6f9f", "color": "#fff", "border": "1px solid #24577d",
    "borderRadius": "4px",
}
_BTN_INACTIVE = {
    "fontSize": "12px", "padding": "4px 10px", "cursor": "pointer",
    "background": "#fff", "color": "#111", "border": "1px solid #aaa",
    "borderRadius": "4px",
}
_SLICE_ROW_BASE = {
    "gap": "24px", "alignItems": "center",
    "padding": "6px 16px", "background": "#f5f5f5", "borderBottom": "1px solid #ddd",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _discover_modes(dataset: str, arch: str) -> list[str]:
    base = PRIOR_DIR / dataset / arch
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / "mean_abs_map.npy").exists()
    )


def _discover_seeds(dataset: str, arch: str, mode: str) -> list[int]:
    base = PRIOR_DIR / dataset / arch / mode
    seeds = []
    for p in base.glob("seed-*_abs_map.npy"):
        try:
            seeds.append(int(p.stem.split("-")[1].split("_")[0]))
        except (IndexError, ValueError):
            pass
    return sorted(seeds)


def _load_region_df(dataset: str, arch: str, mode: str, seed=None) -> pd.DataFrame | None:
    if seed is None:
        p = PRIOR_DIR / dataset / arch / mode / "mean_atlas_regions.csv"
    else:
        p = PRIOR_DIR / dataset / arch / mode / f"seed-{seed}_atlas_regions.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_seed_agreement(dataset: str, arch: str, mode: str) -> pd.DataFrame | None:
    p = PRIOR_DIR / dataset / arch / mode / "seed_agreement.csv"
    return pd.read_csv(p) if p.exists() else None


def _load_comparison(dataset: str, arch: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    comp = PRIOR_DIR / dataset / arch / "comparison"
    vox = pd.read_csv(comp / "voxel_correlations.csv") if (comp / "voxel_correlations.csv").exists() else None
    reg = pd.read_csv(comp / "region_correlations.csv") if (comp / "region_correlations.csv").exists() else None
    return vox, reg


@lru_cache(maxsize=16)
def _group_mask(dataset: str, arch: str, scaler: str = "minmax", channels: str = "t1") -> np.ndarray | None:
    p = PRIOR_DIR / dataset / arch / f"mean_brain_mask_{scaler}_{channels}.npy"
    return np.load(p) if p.exists() else None


@lru_cache(maxsize=64)
def _stat_array(dataset: str, arch: str, mode: str, value_mode: str, seed=None) -> np.ndarray | None:
    """Load attribution volume as float32 array with group mask applied. Cached."""
    base = PRIOR_DIR / dataset / arch / mode
    suffix = "abs_map" if value_mode == "abs" else "signed_map"
    p = base / (f"mean_{suffix}.npy" if seed is None else f"seed-{seed}_{suffix}.npy")
    if not p.exists():
        return None
    arr = np.load(p).astype(np.float32)
    mask = _group_mask(dataset, arch)
    if mask is not None:
        arr = arr.copy()
        arr[~mask] = np.nan
    return arr


def _stat_nii(dataset: str, arch: str, mode: str, value_mode: str, seed=None) -> nib.Nifti1Image | None:
    """Load attribution as NIfTI for nilearn viewer. Tries .nii.gz first for mean."""
    if seed is None:
        fname = "mean_abs_map" if value_mode == "abs" else "mean_signed_map"
        nii_p = PRIOR_DIR / dataset / arch / mode / f"{fname}.nii.gz"
        if nii_p.exists():
            img = nib.load(str(nii_p))
            mask = _group_mask(dataset, arch)
            if mask is not None:
                data = np.asarray(img.dataobj, dtype=np.float32).copy()
                data[~mask] = np.nan
                img = nib.Nifti1Image(data, img.affine)
            return img
    arr = _stat_array(dataset, arch, mode, value_mode, seed)
    return None if arr is None else nib.Nifti1Image(arr, affine=_FS_AFFINE)


@lru_cache(maxsize=16)
def _bg_array(dataset: str, arch: str) -> np.ndarray | None:
    """Cached background T1 numpy array."""
    p = PRIOR_DIR / dataset / arch / "mean_brain_minmax_t1.npy"
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.float32)


def _bg_nii(dataset: str, arch: str) -> nib.Nifti1Image | None:
    arr = _bg_array(dataset, arch)
    return None if arr is None else nib.Nifti1Image(arr, affine=_FS_AFFINE)


@lru_cache(maxsize=32)
def _std_array(dataset: str, arch: str, mode: str) -> np.ndarray | None:
    """Cached std-of-abs-attribution volume (mean-level only, no per-seed variant)."""
    p = PRIOR_DIR / dataset / arch / mode / "std_abs_map.npy"
    if not p.exists():
        return None
    arr = np.log10(np.load(p).astype(np.float32))
    mask = _group_mask(dataset, arch)
    if mask is not None:
        arr = arr.copy()
        arr[~mask] = np.nan
    return arr


@lru_cache(maxsize=8)
def _sigdig_array(dataset: str, arch: str, mode: str) -> np.ndarray | None:
    """Significant decimal digits via verificarlo/significantdigits package.

    Stacks all seed abs maps → significant_digits(stack, axis=0, basis=10).
    Returns float32 array (H, W, D) of decimal digit counts, or None.
    """
    try:
        import significantdigits as sd
    except ImportError:
        return None
    seeds = _discover_seeds(dataset, arch, mode)
    if len(seeds) < 2:
        return None
    base = PRIOR_DIR / dataset / arch / mode
    maps = [
        np.load(base / f"seed-{s}_abs_map.npy").astype(np.float64)
        for s in seeds
        if (base / f"seed-{s}_abs_map.npy").exists()
    ]
    if len(maps) < 2:
        return None
    stack = np.stack(maps, axis=0)  # (n_seeds, H, W, D)
    reference = stack.mean(axis=0)  # (H, W, D) — mean across seeds as reference
    result = sd.significant_digits(stack, reference=reference, axis=0, basis=10)
    # With a scalar/spatial reference, the package removes the seed axis → result is (H, W, D)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mask = _group_mask(dataset, arch)
    if mask is not None:
        result = result.copy()
        result[~mask] = np.nan
    return result


@lru_cache(maxsize=1)
def _atlas_data() -> tuple[np.ndarray, np.ndarray] | None:
    """Return (int32 label volume, affine) for the DKT31 atlas, or None if not found."""
    if not _DKT31_ATLAS_PATH.exists():
        return None
    img = nib.load(str(_DKT31_ATLAS_PATH))
    return np.round(img.get_fdata()).astype(np.int32), img.affine


@lru_cache(maxsize=64)
def _roi_proj_nii(
    dataset: str, arch: str, mode: str, value_mode: str, seed
) -> nib.Nifti1Image | None:
    """Paint region-level stats onto the DKT31 atlas; return NIfTI in MNI space.

    Cached so prefetch and callback share the same result object.
    """
    df = _load_region_df(dataset, arch, mode, seed)
    if df is None or df.empty or "label_id" not in df.columns:
        return None
    col_map = {"abs": "mean_abs", "signed": "mean_signed", "std": "std_abs"}
    col = col_map.get(value_mode)
    if col is None or col not in df.columns:
        return None
    atlas = _atlas_data()
    if atlas is None:
        return None
    atlas_data, affine = atlas
    vol = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        val = float(row[col])
        if np.isfinite(val):
            vol[atlas_data == int(row["label_id"])] = val
    return nib.Nifti1Image(vol, affine=affine)


def _inject_css(html_str: str) -> str:
    """Make nilearn view_img output fill and resize with its containing iframe.

    nilearn's _repr_html_() returns <iframe srcdoc="HTML-ENCODED..."> with BrainSprite
    inside. BrainSprite already has window.addEventListener("resize", t.resize) and sizes
    itself from canvas.parentElement.clientWidth. We decode the srcdoc, inject CSS that
    ties #div_viewer height to 100% and width to height via aspect-ratio, then re-encode.
    """
    import html as _html
    import re as _re

    outer_css = (
        "<style>"
        "html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;"
        "background:#050505;box-sizing:border-box}"
        "body{display:flex;flex-direction:column}"
        "iframe{width:100% !important;height:100% !important;"
        "border:0 !important;display:block;flex:1}"
        "</style>"
    )

    if html_str.strip().startswith("<iframe") and 'srcdoc="' in html_str:
        start_marker = 'srcdoc="'
        si = html_str.find(start_marker)
        if si != -1:
            si += len(start_marker)
            ei = html_str.index('"', si)
            inner_encoded = html_str[si:ei]
            inner = _html.unescape(inner_encoded)

            inner_css = (
                "<style>"
                "html,body{margin:0;padding:0;width:100%;height:100%;"
                "overflow:hidden;background:#000;box-sizing:border-box;"
                "display:flex;flex-direction:column}"
                "#div_viewer{flex:1;min-height:0;max-width:100%;"
                "box-sizing:border-box}"
                "</style>"
            )
            if "</head>" in inner:
                inner = inner.replace("</head>", inner_css + "</head>", 1)

            resize_js = (
                "<script>"
                "window.addEventListener('load',function(){"
                "var c=document.getElementById('3Dviewer');"
                "if(!c||!c.parentElement)return;"
                "var dv=c.parentElement;"
                "if(c.width&&c.height){"
                "dv.style.aspectRatio=(c.width/c.height).toFixed(3);"
                "dv.style.height='100%';"
                "dv.style.width='';"
                "}"
                "});"
                "</script>"
            )
            if "</body>" in inner:
                inner = inner.replace("</body>", resize_js + "</body>", 1)
            else:
                inner += resize_js

            new_encoded = _html.escape(inner)
            html_str = html_str[:si] + new_encoded + html_str[ei:]
            html_str = _re.sub(r'\s+width="\d+"', '', html_str)
            html_str = _re.sub(r'\s+height="\d+"', '', html_str)

    return outer_css + html_str


# ---------------------------------------------------------------------------
# Brain viewers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=_BRAIN_HTML_CACHE_SIZE)
def _make_brain_html(
    dataset: str,
    arch: str,
    mode: str,
    value_mode: str,
    cmap: str,
    threshold_pct: int,
    seed=None,
) -> str:
    from nilearn import plotting

    stat = _stat_nii(dataset, arch, mode, value_mode, seed)
    if stat is None:
        return "<p style='color:grey;padding:20px'>No attribution map found for this selection.</p>"

    nonzero = stat.get_fdata().ravel()
    nonzero = nonzero[np.isfinite(nonzero) & (nonzero != 0)]
    if nonzero.size == 0:
        return "<p style='color:grey;padding:20px'>Attribution map is all zeros.</p>"

    thresh = float(np.percentile(np.abs(nonzero), threshold_pct)) if threshold_pct > 0 else 1e-9
    thresh = thresh if np.isfinite(thresh) and thresh > 0 else 1e-9
    vmax = float(np.abs(nonzero).max())
    symmetric = value_mode == "signed"
    seed_label = f" — seed {seed}" if seed is not None else " — mean"

    bg = _bg_nii(dataset, arch)
    bg_path = None
    if bg is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        nib.save(bg, tmp.name)
        bg_path = tmp.name

    try:
        view = plotting.view_img(
            stat,
            bg_img=bg_path if bg_path else False,
            black_bg=True,
            cmap=cmap,
            symmetric_cmap=symmetric,
            vmax=vmax,
            vmin=None if symmetric else 0.0,
            threshold=thresh,
            colorbar=True,
            title=f"{dataset.upper()} / {arch} / {mode}{seed_label}",
        )
        return _inject_css(view._repr_html_())
    except Exception as exc:
        return f"<p style='color:red;padding:20px'>Brain view error: {exc}</p>"
    finally:
        if bg_path:
            try:
                Path(bg_path).unlink(missing_ok=True)
            except Exception:
                pass


def _prefetch_mode(dataset: str, arch: str, mode: str) -> None:
    """Background: pre-load all seed volumes and std for the current mode into lru_cache."""
    _bg_array(dataset, arch)
    _group_mask(dataset, arch)
    _std_array(dataset, arch, mode)
    seeds = _discover_seeds(dataset, arch, mode)
    for vm in ("abs", "signed"):
        _stat_array(dataset, arch, mode, vm, None)
        for s in seeds:
            _stat_array(dataset, arch, mode, vm, s)
    # _sigdig_array is expensive (stacks all seeds + stats) — triggered on demand only
    # Precompute ROI projection NIfTIs for all value modes (fast: reads CSVs + paints atlas)
    for vm in ("abs", "signed", "std"):
        _roi_proj_nii(dataset, arch, mode, vm, None)
        for s in seeds:
            _roi_proj_nii(dataset, arch, mode, vm, s)


def _make_brain_fig(
    vol: np.ndarray,
    bg: np.ndarray | None,
    cmap: str,
    threshold_pct: int,
    value_mode: str,
    ax_idx: int,
    cor_idx: int,
    sag_idx: int,
    title: str = "",
    rotation: int = 0,
) -> go.Figure:
    _empty = go.Figure().update_layout(
        plot_bgcolor="#111", paper_bgcolor="#111",
        font=dict(color="#ddd"), height=_BRAIN_GRAPH_HEIGHT,
    )

    nonzero = vol[vol != 0]
    if nonzero.size == 0:
        return _empty.update_layout(title="Attribution map is all zeros")

    thresh = float(np.percentile(np.abs(nonzero), threshold_pct)) if threshold_pct > 0 else 0.0
    if value_mode == "signed":
        vmax = float(np.abs(nonzero).max())
        vmin = -vmax
    else:
        vmax = float(nonzero.max())
        vmin = 0.0

    H, W, D = vol.shape
    # FreeSurfer conformed space: dim0(H)=R→L, dim1(W)=S→I (j=0=Superior), dim2(D)=P→A (k=0=Posterior)
    ax_idx  = int(np.clip(ax_idx,  0, W - 1))   # axial:    fix dim 1 (S-I)
    cor_idx = int(np.clip(cor_idx, 0, D - 1))   # coronal:  fix dim 2 (A-P)
    sag_idx = int(np.clip(sag_idx, 0, H - 1))   # sagittal: fix dim 0 (R-L)

    def _rot(arr):
        return np.rot90(arr, k=rotation) if rotation else arr

    # Sagittal vol[sag,:,:] → (W,D): rows=j(S at row-0=top ✓), cols=k(P at col-0=left ✓)
    sag_att = _rot(vol[sag_idx, :, :])
    sag_bg  = _rot(bg[sag_idx, :, :]) if bg is not None else None

    # Axial vol[:,ax,:] → (H,D): rows=i(R-L), cols=k(P-A)
    # .T → (D,H): rows=k(P at top), cols=i(R at left)
    # flipud → A at top; fliplr → L at left, R at right (neurological)
    ax_att = _rot(np.fliplr(np.flipud(vol[:, ax_idx, :].T)))
    ax_bg  = _rot(np.fliplr(np.flipud(bg[:, ax_idx, :].T))) if bg is not None else None

    # Coronal vol[:,:,cor] → (H,W): rows=i(R-L), cols=j(S-I)
    # .T → (W,H): rows=j(S at top ✓), cols=i(R at left)
    # fliplr → L at left, R at right (neurological)
    cor_att = _rot(np.fliplr(vol[:, :, cor_idx].T))
    cor_bg  = _rot(np.fliplr(bg[:, :, cor_idx].T)) if bg is not None else None

    panels = [
        ("Sagittal", sag_att, sag_bg),
        ("Axial",    ax_att,  ax_bg),
        ("Coronal",  cor_att, cor_bg),
    ]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[p[0] for p in panels],
                        horizontal_spacing=0.04)

    for col_i, (_, att_2d, bg_2d) in enumerate(panels, start=1):
        att_masked = att_2d.astype(float).copy()
        att_masked[np.abs(att_2d) < thresh] = np.nan

        if bg_2d is not None:
            bg_f = bg_2d.astype(float)
            bg_range = bg_f.max() - bg_f.min()
            if bg_range > 0:
                bg_f = (bg_f - bg_f.min()) / bg_range
            fig.add_trace(go.Heatmap(
                z=bg_f,
                colorscale="gray",
                showscale=False,
                zmin=0, zmax=1,
                hoverinfo="skip",
            ), row=1, col=col_i)

        fig.add_trace(go.Heatmap(
            z=att_masked,
            coloraxis="coloraxis",
            opacity=0.8 if bg_2d is not None else 1.0,
            hovertemplate="val=%{z:.3e}<extra></extra>",
        ), row=1, col=col_i)

    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        coloraxis=dict(
            colorscale=cmap,
            cmin=vmin, cmax=vmax,
            colorbar=dict(x=1.01, thickness=12,
                          title=dict(text="attr", font=dict(color="#ddd")),
                          tickfont=dict(color="#ddd")),
        ),
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font=dict(color="#ddd"),
        margin=dict(l=10, r=80, t=50, b=10),
        height=_BRAIN_GRAPH_HEIGHT,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Bottom-panel figure builders
# ---------------------------------------------------------------------------

def _make_agreement_fig(dataset: str, arch: str, mode: str) -> go.Figure:
    df = _load_seed_agreement(dataset, arch, mode)
    if df is None:
        return go.Figure().update_layout(
            title="No seed agreement data",
            annotations=[dict(
                text="Run with --n-seeds > 1 to generate seed agreement.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=11, color="grey"),
            )],
        )

    mat = df.values.astype(float)
    off_diag = mat[np.triu_indices(len(mat), k=1)]
    mean_rho = off_diag.mean() if len(off_diag) > 0 else float("nan")
    n = len(df)
    fig = px.imshow(
        mat,
        zmin=0, zmax=1,
        color_continuous_scale="YlOrRd",
        labels={"color": "Spearman ρ"},
        text_auto=".2f",
        title=f"Seed agreement — mean ρ={mean_rho:.3f} (n={n})",
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=50, b=40),
        coloraxis_colorbar=dict(thickness=12),
    )
    return fig


def _make_bar_fig(dataset: str, arch: str, mode: str, value_mode: str, top_n: int, seed=None) -> go.Figure:
    df = _load_region_df(dataset, arch, mode, seed)
    if df is None or df.empty:
        return go.Figure().update_layout(title="No region data")

    _col_map = {"abs": "mean_abs", "signed": "mean_signed", "std": "std_abs"}
    col = _col_map.get(value_mode)
    if col is None or col not in df.columns:
        label = {"sigdig": "sig. digits"}.get(value_mode, value_mode)
        return go.Figure().update_layout(title=f"No region data for {label} map type")

    df = df.dropna(subset=[col]).nlargest(top_n, "mean_abs").sort_values(col, ascending=True)
    err = df["std_abs"].values if "std_abs" in df.columns else None
    colors = ["tomato" if v >= 0 else "steelblue" for v in df[col].values]
    seed_label = f" — seed {seed}" if seed is not None else " — mean"

    fig = go.Figure(go.Bar(
        x=df[col],
        y=df["region"],
        orientation="h",
        error_x=dict(array=err.tolist(), visible=True, color="#888") if err is not None else None,
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>value=%{x:.2e}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Top {top_n} regions — {mode}{seed_label}",
        xaxis_title=f"{'|attribution|' if value_mode == 'abs' else 'signed attribution'}",
        margin=dict(l=max(180, int(df["region"].str.len().max() * 6.5)), r=20, t=50, b=40),
        font=dict(size=10),
        yaxis=dict(tickfont=dict(size=9)),
        height=max(350, top_n * 18),
    )
    return fig


def _make_comparison_table(dataset: str, arch: str) -> go.Figure:
    vox, reg = _load_comparison(dataset, arch)
    if vox is None:
        return go.Figure().update_layout(
            title="No cross-mode comparison data",
            annotations=[dict(
                text="Run experiment with ≥2 modes to generate comparison.",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=11, color="grey"),
            )],
        )

    rows = vox.copy()
    rows["level"] = "voxel"
    if reg is not None:
        r2 = reg.copy()
        r2["level"] = "region"
        rows = pd.concat([rows, r2], ignore_index=True)

    cols = [c for c in ["level", "mode_a", "mode_b", "spearman_rho", "p_value", "n_regions"] if c in rows.columns]
    rows = rows[cols].copy()
    if "n_regions" in rows.columns:
        rows["n_regions"] = rows["n_regions"].where(rows["n_regions"].notna(), "—")

    metric_labels = [c.replace("_", " ").title() for c in cols]
    pair_labels = []
    for _, r in rows.iterrows():
        ma = str(r.get("mode_a", "?"))[:8]
        mb = str(r.get("mode_b", "?"))[:8]
        lv = str(r.get("level", ""))[:1].upper()
        pair_labels.append(f"{lv}:{ma}↔{mb}" if lv else f"{ma}↔{mb}")

    strong = (rows["spearman_rho"].abs() > 0.3).tolist() if "spearman_rho" in rows.columns else [False] * len(rows)
    pair_colors = ["#d4edda" if s else "#fff" for s in strong]

    t_header = ["Metric"] + pair_labels
    t_cells = [metric_labels]
    for i in range(len(rows)):
        col_vals = []
        for c in cols:
            v = rows[c].iloc[i]
            col_vals.append(round(v, 4) if isinstance(v, float) else v)
        t_cells.append(col_vals)

    fig = go.Figure(go.Table(
        header=dict(
            values=t_header,
            fill_color="#f0f2f5",
            font=dict(color="#111", size=10),
            align="left",
        ),
        cells=dict(
            values=t_cells,
            fill_color=["#f0f2f5"] + pair_colors,
            font=dict(color="#111", size=9),
            align="left",
        ),
    ))
    fig.update_layout(
        title=f"Cross-mode Spearman ρ ({dataset.upper()} / {arch})",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    title="Prior Experiment Dashboard",
    suppress_callback_exceptions=True,
)

_initial_dataset = "nki"
_initial_arch = "double_conv"
_initial_modes = _discover_modes(_initial_dataset, _initial_arch)
_initial_mode = _initial_modes[0] if _initial_modes else None
_initial_seeds = _discover_seeds(_initial_dataset, _initial_arch, _initial_mode) if _initial_mode else []

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "background": "#fff"},
    children=[
        dcc.Store(id="cmap-store", data={"abs": _ABS_CMAPS[0], "signed": _SGN_CMAPS[0]}),
        dcc.Store(id="viewer-mode", data="fast"),

        # ---- Header ----
        html.Div(
            style={"padding": "12px 20px", "background": "#f0f2f5", "borderBottom": "1px solid #ddd"},
            children=[html.H2("Prior Experiment Dashboard", style={"margin": 0, "fontSize": "1.2rem"})],
        ),

        # ---- Controls row ----
        html.Div(
            style={
                "display": "flex", "flexWrap": "wrap", "gap": "12px",
                "padding": "10px 16px", "background": "#e8ecf0",
                "alignItems": "center", "borderBottom": "1px solid #ccc",
            },
            children=[
                html.Div([
                    html.Label("Dataset", style={"fontWeight": "bold", "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="dataset-dd",
                        options=[{"label": d.upper(), "value": d} for d in ["nki", "ppmi"]],
                        value=_initial_dataset, clearable=False, style={"width": "110px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Div([
                    html.Label("Architecture", style={"fontWeight": "bold", "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="arch-dd",
                        options=[{"label": a, "value": a} for a in ["double_conv", "cov_pool"]],
                        value=_initial_arch, clearable=False, style={"width": "160px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Div([
                    html.Label("Mode", style={"fontWeight": "bold", "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="mode-dd",
                        options=[{"label": m, "value": m} for m in _initial_modes],
                        value=_initial_mode, clearable=False, style={"width": "220px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Div([
                    html.Label("Seed", style={"fontWeight": "bold", "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="seed-dd",
                        options=(
                            [{"label": "mean", "value": "mean"}]
                            + [{"label": f"seed {i}", "value": str(i)} for i in _initial_seeds]
                        ),
                        value="mean", clearable=False, style={"width": "120px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Div([
                    html.Label("Value", style={"fontWeight": "bold", "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="value-mode-dd",
                        options=[
                            {"label": "Absolute",    "value": "abs"},
                            {"label": "Signed",      "value": "signed"},
                            {"label": "Std (abs)",   "value": "std"},
                            {"label": "Sig. digits", "value": "sigdig"},
                        ],
                        value="abs", clearable=False, style={"width": "130px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Div([
                    html.Label("Colormap", style={"fontWeight": "bold", "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="cmap-dd",
                        options=[{"label": c, "value": c} for c in _ABS_CMAPS],
                        value="hot", clearable=False, style={"width": "130px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Div([
                    html.Label("Template", style={"fontWeight": "bold", "marginRight": "6px"}),
                    dcc.Dropdown(
                        id="template-dd",
                        options=[
                            {"label": "None",     "value": "none"},
                            {"label": "ROI (MNI)", "value": "roi"},
                        ],
                        value="none", clearable=False, style={"width": "120px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
                html.Div(
                    style={"flex": "1", "minWidth": "180px"},
                    children=[
                        html.Label("Threshold percentile", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Slider(
                            id="threshold-slider",
                            min=0, max=95, step=5, value=70,
                            marks={0: "0", 25: "25", 50: "50", 75: "75", 95: "95"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Div(
                    style={"minWidth": "160px"},
                    children=[
                        html.Label("Top-N regions", style={"fontWeight": "bold", "fontSize": "13px"}),
                        dcc.Slider(
                            id="top-n-slider",
                            min=10, max=50, step=5, value=20,
                            marks={10: "10", 30: "30", 50: "50"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                    ],
                ),
                html.Button("⚡ Plotly", id="viewer-btn", n_clicks=0, style=_BTN_ACTIVE),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "8px", "minWidth": "220px"},
                    children=[
                        html.Label("Height:", style={"fontWeight": "bold", "fontSize": "13px", "whiteSpace": "nowrap"}),
                        dcc.Slider(
                            id="brain-height-slider",
                            min=300, max=900, step=60,
                            value=_BRAIN_IFRAME_HEIGHT_DEFAULT,
                            marks={300: "300", 460: "460", 660: "660", 900: "900"},
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="drag",
                            included=False,
                        ),
                    ],
                ),
            ],
        ),

        # ---- Slice sliders (visible in Plotly mode only) ----
        html.Div(
            id="slice-controls",
            style={**_SLICE_ROW_BASE, "display": "flex"},
            children=[
                html.Div([
                    html.Label("Axial z", style={"fontWeight": "bold", "fontSize": "12px", "whiteSpace": "nowrap"}),
                    dcc.Slider(id="ax-slider", min=0, max=255, step=1, value=128,
                               marks={}, tooltip={"placement": "bottom", "always_visible": True},
                               updatemode="drag"),
                ], style={"flex": "1", "minWidth": "200px"}),
                html.Div([
                    html.Label("Coronal y", style={"fontWeight": "bold", "fontSize": "12px", "whiteSpace": "nowrap"}),
                    dcc.Slider(id="cor-slider", min=0, max=255, step=1, value=128,
                               marks={}, tooltip={"placement": "bottom", "always_visible": True},
                               updatemode="drag"),
                ], style={"flex": "1", "minWidth": "200px"}),
                html.Div([
                    html.Label("Sagittal x", style={"fontWeight": "bold", "fontSize": "12px", "whiteSpace": "nowrap"}),
                    dcc.Slider(id="sag-slider", min=0, max=255, step=1, value=128,
                               marks={}, tooltip={"placement": "bottom", "always_visible": True},
                               updatemode="drag"),
                ], style={"flex": "1", "minWidth": "200px"}),
                html.Div([
                    html.Label("Rotation", style={"fontWeight": "bold", "fontSize": "12px", "whiteSpace": "nowrap"}),
                    dcc.RadioItems(
                        id="rotation-ri",
                        options=[{"label": f" {a}°", "value": k} for k, a in [(0, 0), (1, 90), (2, 180), (3, 270)]],
                        value=0,
                        inline=True,
                        style={"fontSize": "12px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px", "minWidth": "240px"}),
            ],
        ),

        # ---- Brain viewer ----
        html.Div(
            style={"padding": "8px 16px"},
            children=[
                html.Div(
                    id="brain-fast-container",
                    style={"display": "block"},
                    children=[dcc.Graph(id="brain-graph", config={"displayModeBar": False})],
                ),
                html.Div(
                    id="brain-slow-container",
                    style={"display": "none"},
                    children=[
                        dcc.Loading(type="circle", children=[
                            html.Iframe(
                                id="brain-iframe",
                                style={
                                    "width": "100%",
                                    "height": f"{_BRAIN_IFRAME_HEIGHT_DEFAULT}px",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "background": "#111",
                                    "display": "block",
                                },
                                srcDoc="<p style='color:grey;padding:20px'>Switch to nilearn mode to load.</p>",
                            ),
                        ]),
                    ],
                ),
                html.Div(
                    id="roi-proj-container",
                    style={"display": "none"},
                    children=[
                        dcc.Loading(type="circle", children=[
                            html.Iframe(
                                id="roi-proj-iframe",
                                srcDoc="",
                                style={
                                    "width": "100%",
                                    "height": f"{_BRAIN_IFRAME_HEIGHT_DEFAULT}px",
                                    "border": "none",
                                    "borderRadius": "4px",
                                    "background": "#111",
                                    "display": "block",
                                },
                            ),
                        ]),
                    ],
                ),
            ],
        ),

        # ---- Bottom panels ----
        html.Div(
            style={"display": "flex", "flexDirection": "column", "gap": "8px", "padding": "0 16px 20px"},
            children=[
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "420px 1fr", "gap": "8px"},
                    children=[
                        html.Div(
                            style={"borderRadius": "4px", "border": "1px solid #eee"},
                            children=[dcc.Graph(id="agreement-graph", config={"displayModeBar": False})],
                        ),
                        html.Div(
                            style={"borderRadius": "4px", "border": "1px solid #eee",
                                   "overflowY": "auto", "maxHeight": "520px"},
                            children=[dcc.Graph(id="bar-graph", config={"displayModeBar": False})],
                        ),
                    ],
                ),
                html.Div(
                    style={"borderRadius": "4px", "border": "1px solid #eee",
                           "overflowY": "auto", "maxHeight": "300px"},
                    children=[dcc.Graph(id="comparison-table", config={"displayModeBar": False})],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("mode-dd", "options"),
    Output("mode-dd", "value"),
    Input("dataset-dd", "value"),
    Input("arch-dd", "value"),
    State("mode-dd", "value"),
)
def update_modes(dataset: str, arch: str, current_mode: str):
    modes = _discover_modes(dataset, arch)
    opts = [{"label": m, "value": m} for m in modes]
    val = current_mode if current_mode in modes else (modes[0] if modes else None)
    return opts, val


@app.callback(
    Output("seed-dd", "options"),
    Output("seed-dd", "value"),
    Input("dataset-dd", "value"),
    Input("arch-dd", "value"),
    Input("mode-dd", "value"),
    State("seed-dd", "value"),
)
def update_seed_dd(dataset: str, arch: str, mode: str, current_seed: str):
    if not mode:
        return [{"label": "mean", "value": "mean"}], "mean"
    seeds = _discover_seeds(dataset, arch, mode)
    opts = [{"label": "mean", "value": "mean"}] + [{"label": f"seed {i}", "value": str(i)} for i in seeds]
    valid = {"mean"} | {str(i) for i in seeds}
    val = current_seed if current_seed in valid else "mean"
    threading.Thread(target=_prefetch_mode, args=(dataset, arch, mode), daemon=True).start()
    return opts, val


@app.callback(
    Output("cmap-store", "data"),
    Input("cmap-dd", "value"),
    State("value-mode-dd", "value"),
    State("cmap-store", "data"),
    prevent_initial_call=True,
)
def save_cmap(cmap: str, value_mode: str, store: dict):
    if cmap is None:
        return store
    store = dict(store)
    # std reuses abs; sigdig has its own key
    key = value_mode if value_mode in ("abs", "signed", "sigdig") else "abs"
    store[key] = cmap
    return store


@app.callback(
    Output("cmap-dd", "options"),
    Output("cmap-dd", "value"),
    Input("value-mode-dd", "value"),
    State("cmap-store", "data"),
)
def update_cmaps(value_mode: str, store: dict):
    store = store or {}
    if value_mode == "signed":
        opts = [{"label": c, "value": c} for c in _SGN_CMAPS]
        remembered = store.get("signed", _SGN_CMAPS[0])
        val = remembered if remembered in _SGN_CMAPS else _SGN_CMAPS[0]
    elif value_mode == "sigdig":
        opts = [{"label": c, "value": c} for c in _SIGDIG_CMAPS]
        remembered = store.get("sigdig", _SIGDIG_CMAPS[0])
        val = remembered if remembered in _SIGDIG_CMAPS else _SIGDIG_CMAPS[0]
    else:  # abs, std
        opts = [{"label": c, "value": c} for c in _ABS_CMAPS]
        remembered = store.get("abs", _ABS_CMAPS[0])
        val = remembered if remembered in _ABS_CMAPS else _ABS_CMAPS[0]
    return opts, val


@app.callback(
    Output("viewer-mode", "data"),
    Output("viewer-btn", "children"),
    Output("viewer-btn", "style"),
    Output("brain-fast-container", "style"),
    Output("brain-slow-container", "style"),
    Output("roi-proj-container", "style"),
    Output("slice-controls", "style"),
    Input("viewer-btn", "n_clicks"),
    Input("template-dd", "value"),
    State("viewer-mode", "data"),
    prevent_initial_call=True,
)
def toggle_viewer(n_clicks, template, current_mode):
    from dash import ctx
    if ctx.triggered_id == "viewer-btn":
        new_mode = "nilearn" if current_mode == "fast" else "fast"
    else:
        new_mode = current_mode or "fast"

    btn_label = "⚡ Plotly" if new_mode == "fast" else "🧠 nilearn"
    btn_style = _BTN_ACTIVE if new_mode == "fast" else _BTN_INACTIVE

    if template == "roi":
        return (new_mode, btn_label, btn_style,
                {"display": "none"}, {"display": "none"}, {"display": "block"},
                {**_SLICE_ROW_BASE, "display": "none"})
    elif new_mode == "fast":
        return (new_mode, btn_label, btn_style,
                {"display": "block"}, {"display": "none"}, {"display": "none"},
                {**_SLICE_ROW_BASE, "display": "flex"})
    else:
        return (new_mode, btn_label, btn_style,
                {"display": "none"}, {"display": "block"}, {"display": "none"},
                {**_SLICE_ROW_BASE, "display": "none"})


@app.callback(
    Output("ax-slider",  "max"), Output("ax-slider",  "value"),
    Output("cor-slider", "max"), Output("cor-slider", "value"),
    Output("sag-slider", "max"), Output("sag-slider", "value"),
    Input("dataset-dd", "value"),
    Input("arch-dd", "value"),
    Input("mode-dd", "value"),
    Input("seed-dd", "value"),
    State("ax-slider",  "value"),
    State("cor-slider", "value"),
    State("sag-slider", "value"),
)
def update_slice_ranges(dataset, arch, mode, seed, cur_ax, cur_cor, cur_sag):
    if not mode:
        return 255, 128, 255, 128, 255, 128
    seed_val = None if seed == "mean" else int(seed) if seed else None
    arr = _stat_array(dataset, arch, mode, "abs", seed_val)
    if arr is None:
        return 255, 128, 255, 128, 255, 128
    H, W, D = arr.shape
    # ax: dim 1 (S-I) → [0, W-1]; cor: dim 2 (A-P) → [0, D-1]; sag: dim 0 (R-L) → [0, H-1]
    ax_v  = int(np.clip(cur_ax  if cur_ax  is not None else W // 2, 0, W - 1))
    cor_v = int(np.clip(cur_cor if cur_cor is not None else D // 2, 0, D - 1))
    sag_v = int(np.clip(cur_sag if cur_sag is not None else H // 2, 0, H - 1))
    return W - 1, ax_v, D - 1, cor_v, H - 1, sag_v


@app.callback(
    Output("brain-graph", "figure"),
    Input("dataset-dd", "value"),
    Input("arch-dd", "value"),
    Input("mode-dd", "value"),
    Input("value-mode-dd", "value"),
    Input("cmap-dd", "value"),
    Input("threshold-slider", "value"),
    Input("seed-dd", "value"),
    Input("ax-slider",  "value"),
    Input("cor-slider", "value"),
    Input("sag-slider", "value"),
    Input("rotation-ri", "value"),
)
def update_brain_fast(dataset, arch, mode, value_mode, cmap, threshold_pct,
                      seed, ax, cor, sag, rotation):
    _dark_empty = go.Figure().update_layout(
        plot_bgcolor="#111", paper_bgcolor="#111",
        font=dict(color="#ddd"), height=_BRAIN_GRAPH_HEIGHT,
    )
    if not mode:
        return _dark_empty
    seed_val = None if seed == "mean" else int(seed) if seed else None
    vm = value_mode or "abs"

    if vm == "std":
        vol = _std_array(dataset, arch, mode)
        map_label = " [std]"
    elif vm == "sigdig":
        try:
            vol = _sigdig_array(dataset, arch, mode)
        except Exception as exc:
            return _dark_empty.update_layout(title=f"Sig. digits error: {type(exc).__name__}: {exc}")
        map_label = " [sig. digits]"
        if vol is None:
            n_seeds = len(_discover_seeds(dataset, arch, mode))
            try:
                import significantdigits  # noqa: F401
                msg = f"Sig. digits: need ≥2 seeds (found {n_seeds})"
            except ImportError:
                msg = "significantdigits package not available (pip install significantdigits)"
            return _dark_empty.update_layout(title=msg)
    else:
        vol = _stat_array(dataset, arch, mode, vm, seed_val)
        map_label = f" — seed {seed_val}" if seed_val is not None else " — mean"

    if vol is None:
        return _dark_empty.update_layout(title="No attribution map found")
    bg = _bg_array(dataset, arch)
    title = f"{dataset.upper()} / {arch} / {mode}{map_label}"
    H, W, D = vol.shape
    # dim 0 (H)=R-L (sag), dim 1 (W)=S-I (ax), dim 2 (D)=P-A (cor)
    ax_i  = int(np.clip(ax  if ax  is not None else W // 2, 0, W - 1))
    cor_i = int(np.clip(cor if cor is not None else D // 2, 0, D - 1))
    sag_i = int(np.clip(sag if sag is not None else H // 2, 0, H - 1))
    # Prefetch the complementary abs/signed so switching is instant
    if vm in ("abs", "signed"):
        other_vm = "signed" if vm == "abs" else "abs"
        threading.Thread(
            target=lambda: _stat_array(dataset, arch, mode, other_vm, seed_val),
            daemon=True,
        ).start()
    # sigdig values can be negative (unreliable voxels) → use signed/symmetric range
    vm_for_fig = "signed" if vm == "sigdig" else ("abs" if vm == "std" else vm)
    return _make_brain_fig(vol, bg, cmap or "hot", threshold_pct or 0,
                           vm_for_fig, ax_i, cor_i, sag_i, title, rotation=rotation or 0)


def _warm_threshold_cache(dataset, arch, mode, value_mode, cmap, skip_t, seed):
    for t in range(0, 100, 5):
        if t != skip_t:
            _make_brain_html(dataset, arch, mode, value_mode, cmap, t, seed)


@app.callback(
    Output("brain-iframe", "srcDoc"),
    Input("dataset-dd", "value"),
    Input("arch-dd", "value"),
    Input("mode-dd", "value"),
    Input("value-mode-dd", "value"),
    Input("cmap-dd", "value"),
    Input("threshold-slider", "value"),
    Input("seed-dd", "value"),
    Input("viewer-mode", "data"),
)
def update_brain_slow(dataset, arch, mode, value_mode, cmap, threshold_pct,
                      seed, viewer_mode):
    if viewer_mode != "nilearn":
        return dash.no_update
    if not mode:
        return "<p style='color:grey;padding:20px'>No mode selected.</p>"
    seed_val = None if seed == "mean" else int(seed) if seed else None
    vm = value_mode or "abs"
    cm = cmap or "hot"
    t  = threshold_pct or 0

    if vm in ("std", "sigdig"):
        arr = _std_array(dataset, arch, mode) if vm == "std" else _sigdig_array(dataset, arch, mode)
        if arr is None:
            return "<p style='color:grey;padding:20px'>No map available for this selection.</p>"
        stat = nib.Nifti1Image(arr, affine=_FS_AFFINE)
        from nilearn import plotting
        flat = arr.ravel()
        nonzero = flat[np.isfinite(flat) & (flat != 0)]
        thresh = float(np.percentile(np.abs(nonzero), t)) if t > 0 and nonzero.size > 0 else 1e-9
        thresh = thresh if np.isfinite(thresh) and thresh > 0 else 1e-9
        vmax = float(nonzero.max()) if nonzero.size > 0 else 1.0
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else 1.0
        try:
            view = plotting.view_img(
                stat, black_bg=True, cmap=cm, symmetric_cmap=False,
                vmax=vmax, vmin=0.0, threshold=thresh, colorbar=True,
                title=f"{dataset.upper()} / {arch} / {mode} [{vm}]",
            )
            return _inject_css(view._repr_html_())
        except Exception as exc:
            return f"<p style='color:red;padding:20px'>Brain view error: {exc}</p>"

    html_str = _make_brain_html(dataset, arch, mode, vm, cm, t, seed_val)
    threading.Thread(
        target=_warm_threshold_cache,
        args=(dataset, arch, mode, vm, cm, t, seed_val),
        daemon=True,
    ).start()
    return html_str


@app.callback(
    Output("roi-proj-iframe", "srcDoc"),
    Input("dataset-dd",    "value"),
    Input("arch-dd",       "value"),
    Input("mode-dd",       "value"),
    Input("value-mode-dd", "value"),
    Input("seed-dd",       "value"),
    Input("cmap-store",    "data"),
    Input("template-dd",   "value"),
)
def update_roi_proj(dataset, arch, mode, value_mode, seed, cmap_store, template):
    if template != "roi" or not all([dataset, arch, mode]):
        return ""
    from nilearn import plotting
    cmap_store = cmap_store or {}
    vm_key = "signed" if value_mode == "signed" else "abs"
    cmap = cmap_store.get(vm_key, "hot")
    seed_val = None if seed in (None, "mean") else int(seed)
    stat = _roi_proj_nii(dataset, arch, mode, value_mode or "abs", seed_val)
    if stat is None:
        return (
            "<p style='color:#aaa;font-family:monospace;padding:8px'>"
            "No ROI data for this mode</p>"
        )
    data = stat.get_fdata()
    finite = data[np.isfinite(data)]
    vmax = float(np.abs(finite).max()) if len(finite) > 0 else 1.0
    symmetric = (value_mode == "signed")
    bg = str(_MNI_T1W_PATH) if _MNI_T1W_PATH.exists() else False
    try:
        view = plotting.view_img(
            stat,
            bg_img=bg,
            black_bg=True,
            cmap=cmap,
            symmetric_cmap=symmetric,
            vmax=vmax,
            vmin=None if symmetric else 0.0,
            colorbar=True,
            title=f"ROI projection — {mode}",
        )
        return _inject_css(view._repr_html_())
    except Exception as exc:
        return f"<p style='color:red;padding:8px'>ROI projection error: {exc}</p>"


@app.callback(
    Output("agreement-graph", "figure"),
    Input("dataset-dd", "value"),
    Input("arch-dd", "value"),
    Input("mode-dd", "value"),
)
def update_agreement(dataset, arch, mode):
    if not mode:
        return go.Figure()
    return _make_agreement_fig(dataset, arch, mode)


@app.callback(
    Output("bar-graph", "figure"),
    Input("dataset-dd", "value"),
    Input("arch-dd", "value"),
    Input("mode-dd", "value"),
    Input("value-mode-dd", "value"),
    Input("top-n-slider", "value"),
    Input("seed-dd", "value"),
)
def update_bar(dataset, arch, mode, value_mode, top_n, seed):
    if not mode:
        return go.Figure()
    seed_val = None if seed == "mean" else int(seed) if seed else None
    return _make_bar_fig(dataset, arch, mode, value_mode or "abs", top_n or 20, seed_val)


@app.callback(
    Output("comparison-table", "figure"),
    Input("dataset-dd", "value"),
    Input("arch-dd", "value"),
)
def update_comparison(dataset, arch):
    return _make_comparison_table(dataset, arch)


_IFRAME_BASE_STYLE = {
    "width": "100%", "border": "none",
    "borderRadius": "4px", "background": "#111", "display": "block",
}

@app.callback(
    Output("brain-iframe",    "style"),
    Output("roi-proj-iframe", "style"),
    Input("brain-height-slider", "value"),
)
def update_brain_height(height_px):
    height_px = int(height_px or _BRAIN_IFRAME_HEIGHT_DEFAULT)
    style = {**_IFRAME_BASE_STYLE, "height": f"{height_px}px"}
    return style, style


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Prior experiment interactive dashboard.")
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Dashboard running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
