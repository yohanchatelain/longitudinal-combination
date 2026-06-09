"""Interactive Voxel Importance Dashboard.

Three-panel dashboard:
  ┌──────────────────────────────────────────────────────────┐
  │  Dataset: [NKI▼]  Aggregation: [mean▼]                  │
  ├──────────────────────────────────────────────────────────┤
  │          nilearn view_img brain viewer (full width)      │
  ├───────────────────────┬──────────────────────────────────┤
  │  mode: plain/         │  mode: plain/dynamic             │
  │    measurements/      │                                  │
  │    dynamic            │                                  │
  │   CNN vs FS ROI       │   Region bar plot (top-30)       │
  └───────────────────────┴──────────────────────────────────┘

Usage:
    python -m brainage_agg.analysis.voxel_importance_dashboard
    # then open http://localhost:8050
"""
from __future__ import annotations

import json
import sys
import tempfile
from functools import lru_cache
from io import StringIO
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr

import dash
from dash import Input, Output, State, dcc, html, ctx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brainage_agg.analysis.roi_to_brain_map import (
    _find_atlas,
    classify_features,
    feature_to_label_id,
    load_lut,
)

from nilearn.datasets import load_mni152_template

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

_DATASETS = {
    "NKI":  ROOT / "brainage_agg" / "outputs",
    "PPMI": ROOT / "brainage_agg" / "ppmi_outputs",
}
_LUT_PATH = ROOT / "FS-LUT.txt"
_ATLAS_PATH = _find_atlas()
_MNI_T1W_PATH = load_mni152_template(resolution=2)

_MEAS_COLORS = {
    "thickness":          "#e377c2",
    "area":               "#17becf",
    "volume":             "#2ca02c",
    "subcortical_volume": "#ff7f0e",
}
_MEAS_SYMBOLS = {
    "thickness":          "circle",
    "area":               "square",
    "volume":             "diamond",
    "subcortical_volume": "cross",
}
_HEMI_SYMBOLS = {
    "left":      "circle",
    "right":     "square",
    "bilateral": "diamond",
}
_GRAPH_CONFIG = {
    "displayModeBar": True,
    "modeBarButtons": [["zoom2d", "pan2d", "resetScale2d"]],
}
_BRAIN_VIEW_WIDTH = 1180
_BRAIN_IFRAME_HEIGHT_DEFAULT = 480
_BRAIN_THRESHOLD_DEFAULT_PERCENTILE = 75
_BRAIN_HTML_CACHE_SIZE = 32

# Load default MNI DKT atlas once at startup
_dkt_atlas_img = nib.load(str(_ATLAS_PATH))
_dkt_atlas_data = np.round(_dkt_atlas_img.get_fdata()).astype(np.int32)
_dkt_atlas_affine = _dkt_atlas_img.affine
_lut = load_lut(_LUT_PATH)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _discover_aggregations(dataset: str) -> list[str]:
    base = _DATASETS[dataset] / "group_diff" / "voxel_importance"
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir()
                  and list(d.glob("region_importance_*.csv")))


def _load_region_importance(dataset: str, aggregation: str) -> pd.DataFrame:
    base = _DATASETS[dataset] / "group_diff" / "voxel_importance" / aggregation
    files = sorted(base.glob("region_importance_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[0])


def _load_cnn_vs_roi(dataset: str, aggregation: str) -> pd.DataFrame:
    base = _DATASETS[dataset] / "group_diff" / "voxel_importance" / aggregation
    files = sorted(base.glob("cnn_vs_roi_comparison_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[0])


def _load_fs_roi(dataset: str, aggregation: str) -> pd.DataFrame:
    path = _DATASETS[dataset] / "group_diff" / f"features_fs_roi_{aggregation}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_brain_atlas(dataset: str) -> dict:
    """Load the parcellation volume used by the brain viewer for a dataset."""
    return {
        "data": _dkt_atlas_data,
        "affine": _dkt_atlas_affine,
        "bg_img": _MNI_T1W_PATH,
        "description": "MNI DKT31",
    }


def _infer_groups(region_df: pd.DataFrame) -> tuple[str, str]:
    """Return (group_a, group_b) by finding the direction on a positive-signed row.

    The direction field encodes 'group_a > group_b' when signed_importance > 0,
    so we read group labels from the first row with positive signed importance.
    """
    pos = region_df[region_df["mean_signed_importance"] > 0]
    for _, row in pos.iterrows():
        d = str(row.get("direction", ""))
        if " > " in d:
            parts = d.split(" > ")
            return parts[0].strip(), parts[1].strip()
    # Fallback: parse any direction row
    for _, row in region_df.iterrows():
        d = str(row.get("direction", ""))
        if " > " in d:
            parts = d.split(" > ")
            return parts[0].strip(), parts[1].strip()
    return ("Group A", "Group B")


def _build_measurements_df(
    dataset: str, aggregation: str, region_df: pd.DataFrame
) -> pd.DataFrame:
    """Build per-measurement scatter data by joining fs_roi with region_importance."""
    fs_roi = _load_fs_roi(dataset, aggregation)
    if fs_roi.empty or region_df.empty:
        return pd.DataFrame()

    groups, atlas_type = classify_features(fs_roi)
    rows = []
    for mtype, features in groups.items():
        for feat in features:
            lid = feature_to_label_id(feat, atlas_type, _lut)
            if lid is None:
                continue
            match = fs_roi[fs_roi["feature"] == feat]
            if match.empty:
                continue
            cohen_d = abs(float(match.iloc[0]["cohen_d"]))
            rows.append({"label_id": lid, "measurement": mtype, "cohen_d": cohen_d})

    if not rows:
        return pd.DataFrame()

    meas_df = pd.DataFrame(rows)
    merged = meas_df.merge(
        region_df[["label_id", "region_name", "hemisphere", "mean_abs_importance"]],
        on="label_id", how="inner",
    )
    return merged


def _region_centroid(
    label_id: int,
    atlas_data: np.ndarray,
    atlas_affine: np.ndarray,
) -> tuple[float, float, float] | None:
    """Return atlas-space coordinates of the centroid of a label."""
    vox = np.argwhere(atlas_data == label_id)
    if len(vox) == 0:
        return None
    centroid_vox = vox.mean(axis=0)
    centroid = (atlas_affine @ np.append(centroid_vox, 1))[:3]
    return tuple(float(c) for c in centroid)


def _inject_responsive_css(html_str: str) -> str:
    """Make nilearn view_img output fill and resize with its containing iframe.

    nilearn's _repr_html_() returns a <iframe srcdoc="HTML-ENCODED..."> element.
    The srcdoc uses BrainSprite, which sizes the canvas from canvas.parentElement.clientWidth.
    BrainSprite already has window.addEventListener("resize", t.resize) built in.
    The trick: make #div_viewer's width track its height via CSS aspect-ratio so that
    when the outer Dash iframe grows taller, div_viewer grows wider, and BrainSprite resizes.
    """
    import html as _html
    import re as _re

    # ── Outer document CSS (targets the nilearn <iframe> element itself) ──────
    outer_css = (
        "<style>"
        "html,body{margin:0;padding:0;width:100%;height:100%;overflow:hidden;"
        "background:#050505;box-sizing:border-box}"
        "body{display:flex;flex-direction:column}"
        # make the nilearn iframe fill the whole Dash iframe
        "iframe{width:100% !important;height:100% !important;"
        "border:0 !important;display:block;flex:1}"
        "</style>"
    )

    # ── Decode and patch the srcdoc ───────────────────────────────────────────
    if html_str.strip().startswith("<iframe") and 'srcdoc="' in html_str:
        start_marker = 'srcdoc="'
        si = html_str.find(start_marker)
        if si != -1:
            si += len(start_marker)
            ei = html_str.index('"', si)   # safe: srcdoc content uses &quot; not "
            inner_encoded = html_str[si:ei]
            inner = _html.unescape(inner_encoded)

            # CSS injected at </head>: make body and #div_viewer fill the viewport
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

            # JS injected at </body>: once BrainSprite has rendered, read the
            # canvas aspect-ratio and set it on #div_viewer so that its width
            # tracks height — BrainSprite's own resize handler does the rest.
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

            # Remove nilearn's fixed width/height HTML attrs on the outer iframe
            new_encoded = _html.escape(inner)
            html_str = (
                html_str[:si] + new_encoded + html_str[ei:]
            )
            html_str = _re.sub(r'\s+width="\d+"', '', html_str)
            html_str = _re.sub(r'\s+height="\d+"', '', html_str)

    # Prepend outer CSS (no </head> in this snippet since it's just an <iframe>)
    return outer_css + html_str


def _make_brain_html(
    dataset: str,
    region_df: pd.DataFrame,
    selected_label_id: int | None = None,
    brain_value_mode: str = "absolute",   # "absolute" | "signed"
    cmap: str = "hot",
    show_t1: bool = True,
    threshold_enabled: bool = False,
    threshold_percentile: int = _BRAIN_THRESHOLD_DEFAULT_PERCENTILE,
) -> str:
    """Paint region_importance onto atlas, call nilearn view_img, return HTML string."""
    from nilearn import plotting

    if region_df.empty:
        return "<p style='color:grey;padding:20px'>No data loaded</p>"

    atlas = _load_brain_atlas(dataset)
    atlas_data = atlas["data"]
    atlas_affine = atlas["affine"]

    metric_col = "mean_abs_importance" if brain_value_mode == "absolute" else "mean_signed_importance"

    vol = np.full(atlas_data.shape, np.nan, dtype=np.float32)
    for _, row in region_df.iterrows():
        lid = int(row["label_id"])
        mask = atlas_data == lid
        if mask.any():
            vol[mask] = float(row[metric_col])

    if np.isfinite(vol).sum() == 0:
        return "<p style='color:grey;padding:20px'>No regions found in atlas</p>"

    cut_coords = None
    finite_vals = vol[np.isfinite(vol)]
    abs_max = float(np.abs(finite_vals).max()) if len(finite_vals) > 0 else 1.0
    threshold_arg = 1e-5  # default small value to avoid zero threshold when enabled
    if threshold_enabled and len(finite_vals) > 0:
        threshold_percentile = int(np.clip(threshold_percentile, 0, 99))
        threshold_arg = float(np.percentile(np.abs(finite_vals), threshold_percentile))

    if selected_label_id is not None:
        centroid = _region_centroid(selected_label_id, atlas_data, atlas_affine)
        if centroid is not None:
            cut_coords = centroid
        mask_sel = atlas_data == selected_label_id
        if mask_sel.any():
            if brain_value_mode == "signed":
                # Preserve sign, boost magnitude
                cur = float(vol[mask_sel].mean()) if np.isfinite(vol[mask_sel]).any() else abs_max
                vol[mask_sel] = np.sign(cur) * abs_max * 2.2 if cur != 0 else abs_max * 2.2
            else:
                vol[mask_sel] = abs_max * 2.2

    stat_img = nib.Nifti1Image(vol, affine=atlas_affine)
    finite_vals = vol[np.isfinite(vol)]
    vmax = float(np.abs(finite_vals).max()) if len(finite_vals) > 0 else 1.0

    bg = atlas["bg_img"] if show_t1 else False
    symmetric = brain_value_mode == "signed"
    vmin_arg = None if symmetric else 0.0

    try:
        view = plotting.view_img(
            stat_img,
            bg_img=bg,
            black_bg=True,
            cut_coords=cut_coords,
            cmap=cmap,
            symmetric_cmap=symmetric,
            vmax=vmax,
            vmin=vmin_arg,
            threshold=threshold_arg,
            colorbar=True,
            title="CNN Voxel Importance",
            width_view=_BRAIN_VIEW_WIDTH,
        )
        return _inject_responsive_css(view._repr_html_())
    except Exception as exc:
        return f"<p style='color:red;padding:20px'>Brain view error: {exc}</p>"


@lru_cache(maxsize=_BRAIN_HTML_CACHE_SIZE)
def _make_brain_html_cached(
    dataset: str,
    region_json: str,
    selected_label_id: int | None,
    brain_value_mode: str,
    cmap: str,
    show_t1: bool,
    threshold_enabled: bool,
    threshold_percentile: int,
) -> str:
    """Cache expensive nilearn HTML generation for repeated view settings."""
    region_df = pd.read_json(StringIO(region_json), orient="split")
    return _make_brain_html(
        dataset,
        region_df,
        selected_label_id=selected_label_id,
        brain_value_mode=brain_value_mode,
        cmap=cmap,
        show_t1=show_t1,
        threshold_enabled=threshold_enabled,
        threshold_percentile=threshold_percentile,
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _agg_options(dataset: str) -> list[dict]:
    return [{"label": a, "value": a} for a in _discover_aggregations(dataset)]


def _make_scatter_plain(
    cnn_roi_df: pd.DataFrame,
    selected_label_id: int | None,
    groups: tuple[str, str],
) -> go.Figure:
    if cnn_roi_df.empty:
        return go.Figure()

    df = cnn_roi_df.copy()
    group_a, group_b = groups

    is_sel = (
        df["label_id"] == selected_label_id
        if selected_label_id is not None
        else pd.Series([False] * len(df))
    )
    df_rest = df[~is_sel]
    df_sel  = df[is_sel]

    traces = [
        go.Scatter(
            x=df_rest["mean_abs_importance"],
            y=df_rest["max_abs_cohen_d_roi"],
            mode="markers+text",
            text=df_rest["region_name"],
            textposition="top center",
            textfont=dict(size=7),
            marker=dict(size=7, color="steelblue", opacity=0.7),
            customdata=df_rest["label_id"].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "CNN |attr|: %{x:.4f}<br>"
                "FS ROI |Cohen d|: %{y:.3f}<extra></extra>"
            ),
            name="regions",
        )
    ]
    if not df_sel.empty:
        traces.append(go.Scatter(
            x=df_sel["mean_abs_importance"],
            y=df_sel["max_abs_cohen_d_roi"],
            mode="markers+text",
            text=df_sel["region_name"],
            textposition="top center",
            textfont=dict(size=8, color="crimson"),
            marker=dict(size=14, color="crimson", symbol="star"),
            customdata=df_sel["label_id"].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "CNN |attr|: %{x:.4f}<br>"
                "FS ROI |Cohen d|: %{y:.3f}<extra></extra>"
            ),
            name="selected",
            showlegend=False,
        ))

    title = f"CNN vs FS ROI — {group_a} vs {group_b}"
    if len(df) >= 3:
        r, _ = pearsonr(df["mean_abs_importance"], df["max_abs_cohen_d_roi"])
        rho, _ = spearmanr(df["mean_abs_importance"], df["max_abs_cohen_d_roi"])
        title += f"<br><sup>Pearson r={r:.2f}, Spearman ρ={rho:.2f}</sup>"

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        xaxis_title="CNN mean |attribution|",
        yaxis_title="FS ROI max |Cohen's d|",
        margin=dict(l=50, r=20, t=60, b=50),
        clickmode="event",
        showlegend=False,
    )
    return fig


def _make_scatter_measurements(
    meas_df: pd.DataFrame,
    selected_label_id: int | None,
    groups: tuple[str, str],
) -> go.Figure:
    if meas_df.empty:
        return go.Figure().update_layout(
            title="No per-measurement data available",
            annotations=[dict(
                text="features_fs_roi CSV not found or no label match<br>"
                     "(check feature-to-atlas label mapping)",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=12, color="grey"),
            )]
        )

    group_a, group_b = groups
    traces = []
    for mtype, color in _MEAS_COLORS.items():
        sub = meas_df[meas_df["measurement"] == mtype]
        if sub.empty:
            continue
        is_sel = sub["label_id"] == selected_label_id if selected_label_id is not None else pd.Series([False]*len(sub), index=sub.index)
        marker_sizes = np.where(is_sel, 14, 8)
        marker_lines = dict(
            width=np.where(is_sel, 2, 0).tolist(),
            color=np.where(is_sel, "crimson", color).tolist(),
        )
        symbols = sub["hemisphere"].map(_HEMI_SYMBOLS).fillna("circle")
        traces.append(go.Scatter(
            x=sub["mean_abs_importance"],
            y=sub["cohen_d"],
            mode="markers",
            name=mtype.replace("_", " "),
            marker=dict(
                size=marker_sizes.tolist(),
                color=color,
                symbol=symbols.tolist(),
                line=marker_lines,
                opacity=0.8,
            ),
            customdata=sub["label_id"].values,
            text=sub["region_name"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"Measurement: {mtype}<br>"
                "CNN |attr|: %{x:.4f}<br>"
                "Cohen's d: %{y:.3f}<extra></extra>"
            ),
        ))

    # Legend for hemisphere shapes
    for hemi, sym in _HEMI_SYMBOLS.items():
        traces.append(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color="grey", symbol=sym),
            name=hemi, showlegend=True,
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"CNN vs FS ROI — {group_a} vs {group_b} (by measurement)",
        xaxis_title="CNN mean |attribution|",
        yaxis_title="|Cohen's d| (per measurement)",
        margin=dict(l=50, r=20, t=60, b=50),
        clickmode="event",
        legend=dict(
            title="<b>Measurement type</b><br><i>Shape = hemisphere</i>",
            font=dict(size=10),
        ),
    )
    return fig


def _make_bar(
    region_df: pd.DataFrame,
    selected_label_id: int | None,
    groups: tuple[str, str],
    top_n: int = 30,
) -> go.Figure:
    if region_df.empty:
        return go.Figure()

    group_a, group_b = groups
    df = region_df.nlargest(top_n, "mean_abs_importance").sort_values(
        "mean_abs_importance", ascending=True
    )

    colors = [
        "tomato" if s > 0 else "steelblue"
        for s in df["mean_signed_importance"]
    ]
    border_colors = [
        "gold" if lid == selected_label_id else c
        for lid, c in zip(df["label_id"], colors)
    ]
    border_widths = [
        3 if lid == selected_label_id else 0
        for lid in df["label_id"]
    ]
    opacities = [
        1.0 if selected_label_id is None or lid == selected_label_id else 0.5
        for lid in df["label_id"]
    ]

    fig = go.Figure(go.Bar(
        x=df["mean_abs_importance"],
        y=df["region_name"],
        orientation="h",
        marker=dict(
            color=colors,
            opacity=opacities,
            line=dict(color=border_colors, width=border_widths),
        ),
        customdata=df["label_id"].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean |attribution|: %{x:.5f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=f"Region importance — top {top_n}",
        xaxis_title="Mean |attribution|",
        yaxis=dict(tickfont=dict(size=9)),
        margin=dict(l=180, r=20, t=50, b=50),
        clickmode="event",
        showlegend=False,
        annotations=[
            dict(x=0.98, y=0.02, xref="paper", yref="paper",
                 text=f"<span style='color:tomato'>■</span> {group_a} > {group_b}"
                      f"  <span style='color:steelblue'>■</span> {group_b} > {group_a}",
                 showarrow=False, align="right", font=dict(size=10)),
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title="Voxel Importance Dashboard")
app.config.suppress_callback_exceptions = True

_initial_dataset = "NKI"
_initial_agg = _discover_aggregations(_initial_dataset)[0] if _discover_aggregations(_initial_dataset) else "mean"

app.layout = html.Div([
    # ── Controls ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Dataset", style={"fontWeight": "bold", "marginRight": "6px"}),
            dcc.Dropdown(
                id="dataset-dd",
                options=[{"label": k, "value": k} for k in _DATASETS],
                value=_initial_dataset,
                clearable=False,
                style={"width": "120px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
        html.Div([
            html.Label("Aggregation", style={"fontWeight": "bold", "marginRight": "6px"}),
            dcc.Dropdown(
                id="agg-dd",
                options=_agg_options(_initial_dataset),
                value=_initial_agg,
                clearable=False,
                style={"width": "180px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
        html.Div(id="ppmi-destrieux-banner", style={"marginLeft": "20px"}),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "20px",
        "padding": "10px 16px", "background": "#f0f2f5",
        "borderBottom": "1px solid #ddd",
    }),

    # ── Brain viewer controls ─────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Colorscale:", style={"fontWeight": "bold", "marginRight": "6px"}),
            dcc.Dropdown(
                id="cmap-dd",
                options=[{"label": c, "value": c} for c in ["hot", "viridis", "plasma", "Oranges"]],
                value="hot",
                clearable=False,
                style={"width": "130px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
        html.Div([
            html.Label("Mode:", style={"fontWeight": "bold", "marginRight": "6px"}),
            dcc.RadioItems(
                id="brain-value-mode",
                options=[
                    {"label": " absolute", "value": "absolute"},
                    {"label": " signed",   "value": "signed"},
                ],
                value="absolute",
                inline=True,
                style={"fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
        dcc.Checklist(
            id="show-t1",
            options=[{"label": " T1 background", "value": "yes"}],
            value=["yes"],
            inline=True,
            style={"fontSize": "13px"},
        ),
        html.Button(
            "Threshold: off",
            id="threshold-btn",
            n_clicks=0,
            style={
                "fontSize": "12px", "padding": "4px 10px",
                "cursor": "pointer", "background": "#fff",
                "border": "1px solid #aaa", "borderRadius": "4px",
            },
        ),
        html.Div([
            html.Label("Quantile", style={"fontWeight": "bold", "fontSize": "13px"}),
            dcc.Slider(
                id="threshold-quantile-slider",
                min=0,
                max=95,
                step=5,
                value=_BRAIN_THRESHOLD_DEFAULT_PERCENTILE,
                marks={0: "0", 25: "25", 50: "50", 75: "75", 95: "95"},
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
                included=False,
            ),
        ], style={"width": "220px", "display": "flex", "alignItems": "center", "gap": "10px"}),
        html.Div([
            html.Label("Height:", style={"fontWeight": "bold", "fontSize": "13px", "whiteSpace": "nowrap"}),
            dcc.Slider(
                id="brain-height-slider",
                min=300,
                max=900,
                step=60,
                value=_BRAIN_IFRAME_HEIGHT_DEFAULT,
                marks={300: "300", 480: "480", 660: "660", 900: "900"},
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
                included=False,
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px", "width": "260px"}),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "24px",
        "padding": "6px 16px", "background": "#e8ecf0",
        "borderBottom": "1px solid #ccc",
    }),

    # ── Brain viewer ──────────────────────────────────────────────────────
    dcc.Loading(
        id="brain-loading",
        type="circle",
        children=html.Iframe(
            id="brain-iframe",
            srcDoc="<p style='color:grey;padding:20px'>Loading…</p>",
            style={"width": "100%", "height": f"{_BRAIN_IFRAME_HEIGHT_DEFAULT}px",
                   "border": "none", "display": "block", "background": "#111"},
        ),
    ),

    # ── Bottom row: scatter + bar ─────────────────────────────────────────
    html.Div([
        # Scatter panel
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Scatter mode:", style={"fontWeight": "bold"}),
                    dcc.RadioItems(
                        id="scatter-mode",
                        options=[
                            {"label": " plain", "value": "plain"},
                            {"label": " measurements", "value": "measurements"},
                            {"label": " dynamic", "value": "dynamic"},
                        ],
                        value="plain",
                        inline=True,
                        style={"fontSize": "13px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                html.Div([
                    html.Button(
                        "✕ Reset",
                        id="scatter-reset-btn",
                        n_clicks=0,
                        style={
                            "fontSize": "12px", "padding": "2px 10px",
                            "cursor": "pointer", "background": "#fff",
                            "border": "1px solid #ccc", "borderRadius": "4px",
                        },
                    ),
                    html.Span(id="scatter-hint",
                              style={"fontSize": "11px", "color": "#666", "marginLeft": "8px"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={
                "display": "flex", "justifyContent": "space-between",
                "alignItems": "center", "padding": "6px 10px",
                "background": "#fafafa", "borderBottom": "1px solid #eee",
            }),
            dcc.Graph(
                id="scatter-graph",
                style={"height": "420px"},
                config=_GRAPH_CONFIG,
            ),
        ], style={"flex": "1", "borderRight": "1px solid #ddd"}),

        # Bar panel
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Bar mode:", style={"fontWeight": "bold"}),
                    dcc.RadioItems(
                        id="bar-mode",
                        options=[
                            {"label": " plain", "value": "plain"},
                            {"label": " dynamic", "value": "dynamic"},
                        ],
                        value="plain",
                        inline=True,
                        style={"fontSize": "13px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                html.Div([
                    html.Button(
                        "✕ Reset",
                        id="bar-reset-btn",
                        n_clicks=0,
                        style={
                            "fontSize": "12px", "padding": "2px 10px",
                            "cursor": "pointer", "background": "#fff",
                            "border": "1px solid #ccc", "borderRadius": "4px",
                        },
                    ),
                    html.Span(id="bar-hint",
                              style={"fontSize": "11px", "color": "#666", "marginLeft": "8px"}),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={
                "display": "flex", "justifyContent": "space-between",
                "alignItems": "center", "padding": "6px 10px",
                "background": "#fafafa", "borderBottom": "1px solid #eee",
            }),
            dcc.Graph(
                id="bar-graph",
                style={"height": "420px"},
                config=_GRAPH_CONFIG,
            ),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "borderTop": "1px solid #ddd"}),

    # ── Hidden state stores ───────────────────────────────────────────────
    dcc.Store(id="region-store"),      # region_importance JSON
    dcc.Store(id="cnn-roi-store"),     # cnn_vs_roi_comparison JSON
    dcc.Store(id="meas-store"),        # per-measurement data JSON
    dcc.Store(id="selected-region"),   # currently selected label_id (int or None)
], style={"fontFamily": "sans-serif", "background": "#fff"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("agg-dd", "options"),
    Output("agg-dd", "value"),
    Input("dataset-dd", "value"),
)
def update_agg_options(dataset):
    opts = _agg_options(dataset)
    val = opts[0]["value"] if opts else None
    return opts, val


@app.callback(
    Output("region-store", "data"),
    Output("cnn-roi-store", "data"),
    Output("selected-region", "data"),
    Output("ppmi-destrieux-banner", "children"),
    Input("dataset-dd", "value"),
    Input("agg-dd", "value"),
)
def load_data(dataset, aggregation):
    if not aggregation:
        return None, None, None, ""
    region_df = _load_region_importance(dataset, aggregation)
    cnn_roi_df = _load_cnn_vs_roi(dataset, aggregation)
    region_json = region_df.to_json(orient="split") if not region_df.empty else None
    cnn_roi_json = cnn_roi_df.to_json(orient="split") if not cnn_roi_df.empty else None
    return region_json, cnn_roi_json, None, ""


@app.callback(
    Output("meas-store", "data"),
    Input("dataset-dd", "value"),
    Input("agg-dd", "value"),
    Input("scatter-mode", "value"),
    State("region-store", "data"),
)
def load_measurements(dataset, aggregation, scatter_mode, region_json):
    if scatter_mode != "measurements" or not aggregation or not region_json:
        return None
    region_df = pd.read_json(StringIO(region_json), orient="split")
    meas_df = _build_measurements_df(dataset, aggregation, region_df)
    return meas_df.to_json(orient="split") if not meas_df.empty else None


@app.callback(
    Output("cmap-dd", "options"),
    Output("cmap-dd", "value"),
    Input("brain-value-mode", "value"),
)
def update_cmap_options(brain_value_mode):
    if brain_value_mode == "signed":
        opts = ["RdBu_r", "coolwarm", "bwr", "PiYG"]
    else:
        opts = ["hot", "viridis", "plasma", "Oranges"]
    return [{"label": c, "value": c} for c in opts], opts[0]


@app.callback(
    Output("threshold-btn", "children"),
    Output("threshold-btn", "style"),
    Input("threshold-btn", "n_clicks"),
    Input("threshold-quantile-slider", "value"),
)
def update_threshold_button(n_clicks, threshold_percentile):
    enabled = bool(n_clicks and n_clicks % 2)
    threshold_percentile = int(threshold_percentile or _BRAIN_THRESHOLD_DEFAULT_PERCENTILE)
    style = {
        "fontSize": "12px", "padding": "4px 10px",
        "cursor": "pointer", "borderRadius": "4px",
    }
    if enabled:
        style.update({
            "background": "#2f6f9f", "color": "#fff",
            "border": "1px solid #24577d",
        })
        return f"Keep top {100 - threshold_percentile}%", style
    style.update({
        "background": "#fff", "color": "#111",
        "border": "1px solid #aaa",
    })
    return "Threshold: off", style


@app.callback(
    Output("brain-iframe", "srcDoc"),
    Input("dataset-dd", "value"),
    Input("region-store", "data"),
    Input("selected-region", "data"),
    Input("brain-value-mode", "value"),
    Input("cmap-dd", "value"),
    Input("show-t1", "value"),
    Input("threshold-btn", "n_clicks"),
    Input("threshold-quantile-slider", "value"),
)
def update_brain(
    dataset,
    region_json,
    selected_label_id,
    brain_value_mode,
    cmap,
    show_t1_val,
    threshold_clicks,
    threshold_percentile,
):
    if not region_json:
        return "<p style='color:grey;padding:20px'>No data — select a dataset and aggregation.</p>"
    selected_label_id = int(selected_label_id) if selected_label_id is not None else None
    brain_value_mode = brain_value_mode or "absolute"
    cmap = cmap or "hot"
    show_t1 = bool(show_t1_val)  # checklist returns list; truthy if non-empty
    threshold_enabled = bool(threshold_clicks and threshold_clicks % 2)
    threshold_percentile = int(threshold_percentile or _BRAIN_THRESHOLD_DEFAULT_PERCENTILE)
    return _make_brain_html_cached(
        dataset,
        region_json,
        selected_label_id,
        brain_value_mode,
        cmap,
        show_t1,
        threshold_enabled,
        threshold_percentile,
    )


@app.callback(
    Output("scatter-graph", "figure"),
    Input("cnn-roi-store", "data"),
    Input("meas-store", "data"),
    Input("scatter-mode", "value"),
    Input("selected-region", "data"),
    State("region-store", "data"),
)
def update_scatter(cnn_roi_json, meas_json, scatter_mode, selected_label_id, region_json):
    groups = ("Group A", "Group B")
    if region_json:
        region_df = pd.read_json(StringIO(region_json), orient="split")
        groups = _infer_groups(region_df)

    if scatter_mode == "measurements":
        if not meas_json:
            return _make_scatter_measurements(pd.DataFrame(), selected_label_id, groups)
        meas_df = pd.read_json(StringIO(meas_json), orient="split")
        return _make_scatter_measurements(meas_df, selected_label_id, groups)

    if not cnn_roi_json:
        return go.Figure()
    cnn_roi_df = pd.read_json(StringIO(cnn_roi_json), orient="split")
    return _make_scatter_plain(cnn_roi_df, selected_label_id, groups)


@app.callback(
    Output("bar-graph", "figure"),
    Input("region-store", "data"),
    Input("bar-mode", "value"),
    Input("selected-region", "data"),
)
def update_bar(region_json, bar_mode, selected_label_id):
    if not region_json:
        return go.Figure()
    region_df = pd.read_json(StringIO(region_json), orient="split")
    groups = _infer_groups(region_df)
    sel = selected_label_id if bar_mode == "dynamic" else None
    return _make_bar(region_df, sel, groups)


@app.callback(
    Output("selected-region", "data", allow_duplicate=True),
    Input("scatter-graph", "clickData"),
    Input("bar-graph", "clickData"),
    Input("scatter-reset-btn", "n_clicks"),
    Input("bar-reset-btn", "n_clicks"),
    State("scatter-mode", "value"),
    State("bar-mode", "value"),
    State("selected-region", "data"),
    prevent_initial_call=True,
)
def update_selection(scatter_click, bar_click, _sr, _br, scatter_mode, bar_mode, current_sel):
    triggered = ctx.triggered_id
    if triggered in ("scatter-reset-btn", "bar-reset-btn"):
        return None
    if triggered == "scatter-graph" and scatter_mode == "dynamic" and scatter_click:
        pts = scatter_click.get("points", [])
        if pts:
            lid = pts[0].get("customdata")
            return None if lid == current_sel else lid
    if triggered == "bar-graph" and bar_mode == "dynamic" and bar_click:
        pts = bar_click.get("points", [])
        if pts:
            lid = pts[0].get("customdata")
            return None if lid == current_sel else lid
    return current_sel


@app.callback(
    Output("scatter-hint", "children"),
    Input("scatter-mode", "value"),
)
def update_scatter_hint(scatter_mode):
    if scatter_mode == "dynamic":
        return "💡 Click a point to select it — click again to deselect"
    return ""


@app.callback(
    Output("bar-hint", "children"),
    Input("bar-mode", "value"),
)
def update_bar_hint(bar_mode):
    if bar_mode == "dynamic":
        return "💡 Click a bar to select it — click again to deselect"
    return ""


@app.callback(
    Output("brain-iframe", "style"),
    Input("brain-height-slider", "value"),
)
def update_brain_height(height_px):
    height_px = int(height_px or _BRAIN_IFRAME_HEIGHT_DEFAULT)
    return {"width": "100%", "height": f"{height_px}px",
            "border": "none", "display": "block", "background": "#111"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"Dashboard running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
