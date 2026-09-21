"""Build and save evaluation figures."""

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

_PLOT_FONT_FAMILY = "Avenir Next Heavy"
_PLOT_TEXT_FONT_FAMILY = "Avenir Next Medium"
MATPLOTLIB_PAPER_FONT_FAMILIES = ["Avenir Next", "Avenir", "DejaVu Sans"]
TARGET_STRING_LANG_ORDER = [
    "ar",
    "bg",
    "cs",
    "de",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "gl",
    "hi",
    "id",
    "is",
    "it",
    "ja",
    "ko",
    "mr",
    "pl",
    "pt_br",
    "ru",
    "sr",
    "sv",
    "th",
    "tr",
    "uk",
    "ur",
    "zh",
]


def _build_target_string_lang2color() -> dict[str, Any]:
    palette = list(plt.colormaps.get_cmap("tab20").colors)
    palette.extend(plt.colormaps.get_cmap("tab20b").colors)
    return {
        lang: palette[idx % len(palette)]
        for idx, lang in enumerate(TARGET_STRING_LANG_ORDER)
    }


TARGET_STRING_LANG2COLOR = _build_target_string_lang2color()


################################################################################
# Shared figure output and styling
################################################################################

def set_matplotlib_paper_font(
    families: Sequence[str] = tuple(MATPLOTLIB_PAPER_FONT_FAMILIES),
) -> None:
    """Use paper-facing sans-serif fonts for Matplotlib figures."""
    font_dir = Path(__file__).resolve().parent / ".cache" / "fonts"
    for font_path in (
        font_dir / "AvenirNext-Medium.ttf",
        font_dir / "AvenirNext-DemiBold.ttf",
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(families),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_matplotlib_figure_bundle(
    fig,
    stem_path: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """Save a matplotlib figure under a stable stem in multiple formats."""
    stem_path = Path(stem_path)
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem_path.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches=bbox_inches)


def _set_split_figure_title(
    fig,
    title: str | None,
    *,
    title_fontsize: int = 13,
    title_fontweight: str | int = "normal",
    subtitle_fontsize: int = 10,
    subtitle_linespacing: float = 1.0,
    title_y: float = 0.995,
    subtitle_y: float = 0.958,
) -> None:
    if not title:
        return
    lines = str(title).splitlines()
    main_title = lines[0]
    subtitle = "\n".join(line for line in lines[1:] if line.strip())
    fig.suptitle(
        main_title,
        fontsize=title_fontsize,
        fontfamily="Avenir Next",
        fontweight=title_fontweight,
        y=title_y,
    )
    if subtitle:
        fig.text(
            0.5,
            subtitle_y,
            subtitle,
            ha="center",
            va="top",
            fontsize=subtitle_fontsize,
            linespacing=subtitle_linespacing,
        )


################################################################################
# Target-string figures
################################################################################

def _target_string_lang_colors(languages: Sequence[str]) -> dict[str, Any]:
    languages = list(dict.fromkeys(languages))
    fallback = list(plt.colormaps.get_cmap("tab20c").colors)
    colors = {}
    fallback_idx = 0
    for lang in languages:
        if lang in TARGET_STRING_LANG2COLOR:
            colors[lang] = TARGET_STRING_LANG2COLOR[lang]
        else:
            colors[lang] = fallback[fallback_idx % len(fallback)]
            fallback_idx += 1
    return colors


def _target_data_source_for_lang(lang: str) -> str:
    return f"translation_to_{str(lang).replace('-', '_')}"


LANGUAGE_DISPLAY_NAMES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "cs": "Czech",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "hi": "Hindi",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mr": "Marathi",
    "pl": "Polish",
    "pt_br": "Portuguese (Brazil)",
    "ru": "Russian",
    "sr": "Serbian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "zh": "Chinese",
}


def _format_lang_label(lang: Any) -> str:
    if pd.isna(lang):
        return ""
    lang_key = str(lang)
    return LANGUAGE_DISPLAY_NAMES.get(lang_key, lang_key.replace("_", "-"))


def _ordered_present(values: Iterable[Any], preferred: Sequence[Any] | None = None) -> list[Any]:
    present = [value for value in values if pd.notna(value)]
    present_set = set(present)
    if preferred is not None:
        ordered = [value for value in preferred if value in present_set]
        ordered.extend(sorted(value for value in present_set if value not in set(ordered)))
        return ordered
    return sorted(present_set)


def _coerce_plot_layers(df: pd.DataFrame, layer_col: str) -> list[float]:
    if df.empty or layer_col not in df.columns:
        return []
    return [float(value) for value in sorted(df[layer_col].dropna().unique())]


def _axis_label_for_layer_col(layer_col: str) -> str:
    if layer_col == "layer_norm":
        return "relative layer depth"
    if layer_col == "layer_bin":
        return "normalized layer bin"
    if layer_col == "layer":
        return "layer"
    return layer_col


def _layer_axis_limits(layer_values: Sequence[float]) -> tuple[float, float]:
    if not layer_values:
        return (0.0, 1.0)
    lo = min(layer_values)
    hi = max(layer_values)
    if np.isclose(lo, hi):
        return (lo - 0.5, hi + 0.5)
    span = hi - lo
    margin = 0.02 if span <= 1.1 else 0.5
    return (lo - margin, hi + margin)


def _entropy_limits(
    prompt_df: pd.DataFrame,
    *,
    data_sources: Sequence[str],
    models: Sequence[str],
    layer_col: str,
    prompt_langs: Sequence[str] | None = None,
) -> dict[str, tuple[float | None, float | None]]:
    limits = {}
    for col in ["entropy", "tgt_first_token_vocab_entropy_bits"]:
        if col not in prompt_df.columns or prompt_df.empty:
            limits[col] = (None, None)
            continue
        df = prompt_df[
            prompt_df["data_source"].isin(list(data_sources))
            & prompt_df["display_model_name"].isin(list(models))
        ]
        if prompt_langs is not None and "prompt_lang" in df.columns:
            df = df[df["prompt_lang"].isin(list(prompt_langs))]
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            limits[col] = (None, None)
        else:
            low = float(values.min())
            high = float(values.max())
            if np.isclose(low, high):
                high = low + 1e-9
            limits[col] = (low, high)
    return limits


def _add_entropy_colorbars(
    fig,
    *,
    entropy_limits: Mapping[str, tuple[float | None, float | None]],
    right: float = 0.90,
    lang_top: float | None = None,
    vocab_bottom: float | None = None,
) -> None:
    colorbar_height = 0.24
    lang_bottom = 0.58 if lang_top is None else lang_top - colorbar_height
    vocab_bottom = 0.26 if vocab_bottom is None else vocab_bottom
    specs = [
        ("entropy", "Lang Entrp", "viridis", [right + 0.012, lang_bottom, 0.012, colorbar_height]),
        ("tgt_first_token_vocab_entropy_bits", "Vocab Entrp", "magma", [right + 0.012, vocab_bottom, 0.012, colorbar_height]),
    ]
    for col, label, cmap, rect in specs:
        vmin, vmax = entropy_limits.get(col, (None, None))
        if vmin is None or vmax is None:
            continue
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cax = fig.add_axes(rect)
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(label, fontsize=11)
        cb.ax.tick_params(labelsize=10)


def _add_model_row_label(ax, model: str) -> None:
    ax.annotate(
        model,
        xy=(-0.4, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        fontsize=14,
        fontfamily="Avenir Next",
        fontweight=600,
        annotation_clip=False,
    )


def _aggregate_entropy(
    prompt_df: pd.DataFrame,
    *,
    data_source: str,
    display_model_name: str,
    layer_col: str,
    prompt_lang: str | None = None,
    prompt_langs: Sequence[str] | None = None,
) -> pd.DataFrame:
    if prompt_df.empty or layer_col not in prompt_df.columns:
        return pd.DataFrame(columns=[layer_col, "entropy", "tgt_first_token_vocab_entropy_bits"])
    df = prompt_df[
        (prompt_df["data_source"] == data_source)
        & (prompt_df["display_model_name"] == display_model_name)
    ].copy()
    if prompt_lang is not None and "prompt_lang" in df.columns:
        df = df[df["prompt_lang"] == prompt_lang]
    if prompt_langs is not None and "prompt_lang" in df.columns:
        df = df[df["prompt_lang"].isin(list(prompt_langs))]
    cols = [layer_col]
    for col in ["entropy", "tgt_first_token_vocab_entropy_bits"]:
        if col in df.columns:
            cols.append(col)
    if len(cols) == 1 or df.empty:
        return pd.DataFrame(columns=[layer_col, "entropy", "tgt_first_token_vocab_entropy_bits"])
    return df[cols].groupby(layer_col, as_index=False).mean(numeric_only=True)


def _plot_entropy_strip(
    ax,
    entropy_df: pd.DataFrame,
    *,
    layer_values: Sequence[int],
    layer_col: str,
    value_col: str,
    label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    show_label: bool = False,
) -> None:
    ax.set_yticks([])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    if not layer_values:
        ax.set_axis_off()
        return

    x_min, x_max = _layer_axis_limits(layer_values)
    values = pd.Series(index=list(layer_values), dtype=float)
    if value_col in entropy_df.columns and not entropy_df.empty:
        keyed = entropy_df.set_index(layer_col)[value_col]
        values.loc[keyed.index.intersection(values.index)] = keyed.loc[keyed.index.intersection(values.index)].astype(float)

    if values.notna().any():
        xs = np.asarray(values.index.to_list(), dtype=float)
        if len(xs) == 1:
            edges = np.asarray([xs[0] - 0.5, xs[0] + 0.5], dtype=float)
        else:
            mids = (xs[:-1] + xs[1:]) / 2.0
            edges = np.concatenate(
                [
                    [xs[0] - (mids[0] - xs[0])],
                    mids,
                    [xs[-1] + (xs[-1] - mids[-1])],
                ]
            )
        ax.pcolormesh(
            edges,
            [0, 1],
            values.to_numpy(dtype=float)[None, :],
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        ax.set_axis_off()
        return
    if show_label:
        ax.set_ylabel(
            label,
            rotation=0,
            ha="right",
            va="center",
            fontsize=11,
            fontfamily="Avenir",
            fontweight=400,
            labelpad=10,
            y=1.0,
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(x_min, x_max)


def _line_panel(
    ax,
    panel_df: pd.DataFrame,
    *,
    layer_col: str,
    value_col: str,
    target_langs: Sequence[str],
    colors: Mapping[str, Any],
    bold_lang: str | None = None,
    yscale: str = "log",
    ylim: tuple[float, float] | None = None,
) -> set[str]:
    plotted = set()
    if panel_df.empty:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=11, color="#777777", transform=ax.transAxes)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=11)
        return plotted

    if value_col in panel_df.columns and not panel_df.empty:
        numeric_values = pd.to_numeric(panel_df[value_col], errors="coerce")
        peak_df = panel_df.assign(_plot_value=numeric_values)
        peak_df = peak_df[np.isfinite(peak_df["_plot_value"])]
        peak_scores = peak_df.groupby("tgt_lang", dropna=False)["_plot_value"].max().sort_values(ascending=False)
    else:
        peak_scores = pd.Series(dtype=float)
    peak_langs = set(peak_scores.head(3).index.tolist())
    bold_langs = set(peak_langs)
    if bold_lang is not None and bold_lang in set(panel_df["tgt_lang"].dropna()):
        bold_langs.add(bold_lang)

    for tgt_lang in target_langs:
        series = panel_df[panel_df["tgt_lang"] == tgt_lang].sort_values(layer_col)
        if series.empty:
            continue
        x = series[layer_col].to_numpy()
        y = series[value_col].to_numpy(dtype=float)
        if yscale == "log":
            mask = np.isfinite(y) & (y > 0)
            x = x[mask]
            y = y[mask]
            if len(y) == 0:
                continue
        else:
            mask = np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if len(y) == 0:
                continue
        is_self = bold_lang is not None and tgt_lang == bold_lang
        is_peak_lang = tgt_lang in bold_langs
        is_peak_other = is_peak_lang and not is_self
        ax.plot(
            x,
            y,
            color=colors.get(tgt_lang, "black"),
            linewidth=2.3 if is_self else (1.9 if is_peak_lang else 0.9),
            alpha=0.98 if (is_self or is_peak_lang) else 0.45,
            linestyle="--" if is_peak_other else "-",
            label=_format_lang_label(tgt_lang),
        )
        plotted.add(tgt_lang)

    if yscale:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="both", linewidth=0.4, alpha=0.22)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=11)
    return plotted


_CONTROLLED_STARTW_GRID_BOTTOM = 0.30


def _add_controlled_startw_bottom_layout(
    fig,
    *,
    language_handles: Sequence[Any],
    style_handles: Sequence[Any],
    layer_label: str,
    top: float,
    left: float = 0.17,
    right: float = 0.90,
) -> None:
    """Add the controlled Start(w) x-label and legends below the grid."""
    grid_bottom = _CONTROLLED_STARTW_GRID_BOTTOM
    layer_y = grid_bottom - 0.06
    language_legend_center_x = 0.5
    language_legend_y = 0.065
    style_legend_y = 0.025

    fig.subplots_adjust(left=left, right=right, bottom=grid_bottom, top=top)
    fig.supxlabel(layer_label, fontsize=15, y=layer_y)
    if language_handles:
        lang_legend = fig.legend(
            handles=language_handles,
            loc="lower center",
            bbox_to_anchor=(language_legend_center_x, language_legend_y),
            ncol=min(9, len(language_handles)),
            frameon=False,
            fontsize=13,
            title="Candidate Latent Languages:",
            title_fontsize=13,
            handlelength=2.0,
            columnspacing=0.95,
        )
        lang_legend.get_title().set_fontfamily("Avenir Next")
        lang_legend.get_title().set_fontweight(600)
        fig.add_artist(lang_legend)
    fig.legend(
        handles=style_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, style_legend_y),
        ncol=2,
        frameon=False,
        fontsize=13,
    )


def plot_startw_copy_cloze_grid(
    menu_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    *,
    data_source: str,
    models: Sequence[str],
    prompt_langs: Sequence[str],
    target_langs: Sequence[str] | None = None,
    layer_col: str = "layer",
    value_col: str = "menu_mean_prob_share",
    yscale: str = "log",
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    highlight_lang: str | None = None,
    style_match_label: str = "latent lang == prompt lang",
    ylabel: str = "Start(w) probability per latent language",
):
    """Plot a copy/cloze Start(w) grid with aligned vocab-entropy strips."""
    df = menu_df[
        (menu_df["data_source"] == data_source)
        & (menu_df["display_model_name"].isin(list(models)))
        & (menu_df["prompt_lang"].isin(list(prompt_langs)))
    ].copy()
    if target_langs is not None:
        df = df[df["tgt_lang"].isin(list(target_langs))]
    prompt_langs = _ordered_present(df["prompt_lang"].unique(), prompt_langs)
    target_langs = _ordered_present(df["tgt_lang"].unique(), target_langs)
    colors = _target_string_lang_colors(target_langs)
    entropy_limits = _entropy_limits(
        prompt_df,
        data_sources=[data_source],
        models=models,
        prompt_langs=prompt_langs,
        layer_col=layer_col,
    )

    agg = (
        df.groupby(["display_model_name", "prompt_lang", "tgt_lang", layer_col], dropna=False)[value_col]
        .mean()
        .reset_index()
        if not df.empty
        else df
    )

    figsize = figsize or (max(13, 2.3 * len(prompt_langs)), max(5.1, 2.55 * len(models)))
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        len(models),
        len(prompt_langs),
        hspace=0.24,
        wspace=0.08,
    )
    axes: list[list[tuple[Any, Any, Any]]] = []
    for row_idx in range(len(models)):
        row_axes = []
        for col_idx in range(len(prompt_langs)):
            inner = outer[row_idx, col_idx].subgridspec(
                3,
                1,
                height_ratios=[0.36, 0.36, 1.85],
                hspace=0.04,
            )
            row_axes.append(
                (
                    fig.add_subplot(inner[0, 0]),
                    fig.add_subplot(inner[1, 0]),
                    fig.add_subplot(inner[2, 0]),
                )
            )
        axes.append(row_axes)

    plotted_langs = set()
    for row_idx, model in enumerate(models):
        row_layers = _coerce_plot_layers(df[df["display_model_name"] == model], layer_col)
        row_x_min, row_x_max = _layer_axis_limits(row_layers)
        for col_idx, prompt_lang in enumerate(prompt_langs):
            strip_lang_ax, strip_vocab_ax, line_ax = axes[row_idx][col_idx]

            entropy_df = _aggregate_entropy(
                prompt_df,
                data_source=data_source,
                display_model_name=model,
                prompt_lang=prompt_lang,
                layer_col=layer_col,
            )
            _plot_entropy_strip(
                strip_lang_ax,
                entropy_df,
                layer_values=row_layers,
                layer_col=layer_col,
                value_col="entropy",
                label="Lang Entrp",
                cmap="viridis",
                vmin=entropy_limits["entropy"][0],
                vmax=entropy_limits["entropy"][1],
                show_label=col_idx == 0,
            )
            _plot_entropy_strip(
                strip_vocab_ax,
                entropy_df,
                layer_values=row_layers,
                layer_col=layer_col,
                value_col="tgt_first_token_vocab_entropy_bits",
                label="Vocab Entrp",
                cmap="magma",
                vmin=entropy_limits["tgt_first_token_vocab_entropy_bits"][0],
                vmax=entropy_limits["tgt_first_token_vocab_entropy_bits"][1],
                show_label=col_idx == 0,
            )

            panel = agg[
                (agg["display_model_name"] == model)
                & (agg["prompt_lang"] == prompt_lang)
            ]
            plotted_langs.update(
                _line_panel(
                    line_ax,
                    panel,
                    layer_col=layer_col,
                    value_col=value_col,
                    target_langs=target_langs,
                    colors=colors,
                    bold_lang=highlight_lang or prompt_lang,
                    yscale=yscale,
                    ylim=ylim,
                )
            )
            if row_idx == 0:
                strip_lang_ax.set_title(
                    _format_lang_label(prompt_lang),
                    fontsize=14,
                    fontfamily="Avenir Next",
                    fontweight=600,
                    pad=6,
                )
            if col_idx == 0:
                _add_model_row_label(line_ax, model)
            line_ax.set_xlim(row_x_min, row_x_max)
            if layer_col == "layer":
                line_ax.xaxis.set_major_locator(MultipleLocator(10))
            strip_lang_ax.set_xlim(row_x_min, row_x_max)
            strip_vocab_ax.set_xlim(row_x_min, row_x_max)
            line_ax.tick_params(axis="x", labelbottom=True, pad=1)
            line_ax.tick_params(axis="y", left=col_idx == 0, labelleft=col_idx == 0)

    _set_split_figure_title(fig, title, title_fontsize=16, title_fontweight=600, subtitle_fontsize=12)
    grid_top = 0.865 if title else 0.92
    fig.supylabel(ylabel, fontsize=15, x=0.075, y=(_CONTROLLED_STARTW_GRID_BOTTOM + grid_top) / 2)

    handles = [
        plt.Line2D([0], [0], color=colors[lang], lw=2.0, label=_format_lang_label(lang))
        for lang in target_langs
        if lang in plotted_langs
    ]
    style_handles = [
        plt.Line2D([0], [0], color="#222222", lw=2.6, linestyle="-", label=style_match_label),
        plt.Line2D([0], [0], color="#222222", lw=2.0, linestyle="--", label="top-3 peak langs if not column lang"),
    ]
    _add_controlled_startw_bottom_layout(
        fig,
        language_handles=handles,
        style_handles=style_handles,
        layer_label="Layer" if layer_col == "layer" else _axis_label_for_layer_col(layer_col),
        top=grid_top,
    )
    _add_entropy_colorbars(
        fig,
        entropy_limits=entropy_limits,
        right=0.90,
        lang_top=axes[0][0][2].get_position().y1,
        vocab_bottom=axes[-1][0][2].get_position().y0,
    )
    return fig

def plot_startw_translation_target_grid(
    menu_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    *,
    models: Sequence[str],
    target_data_langs: Sequence[str],
    prompt_langs: Sequence[str] | None = None,
    target_langs: Sequence[str] | None = None,
    layer_col: str = "layer",
    value_col: str = "menu_mean_prob_share",
    yscale: str = "log",
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ylabel: str = "Start(w) probability per latent language",
    grid_left: float = 0.17,
    grid_right: float = 0.90,
    grid_wspace: float = 0.08,
    ylabel_x: float = 0.075,
):
    """Static translation-to-target grid; columns are requested translation targets."""
    data_sources = [_target_data_source_for_lang(lang) for lang in target_data_langs]
    df = menu_df[
        (menu_df["display_model_name"].isin(list(models)))
        & (menu_df["data_source"].isin(data_sources))
    ].copy()
    if prompt_langs is not None:
        df = df[df["prompt_lang"].isin(list(prompt_langs))]
    if target_langs is not None:
        df = df[df["tgt_lang"].isin(list(target_langs))]
    present_data_sources = set(df["data_source"].dropna())
    target_data_langs = [
        lang for lang in target_data_langs if _target_data_source_for_lang(lang) in present_data_sources
    ]
    target_langs = _ordered_present(df["tgt_lang"].unique(), target_langs)
    colors = _target_string_lang_colors(target_langs)
    entropy_limits = _entropy_limits(
        prompt_df,
        data_sources=data_sources,
        models=models,
        prompt_langs=prompt_langs,
        layer_col=layer_col,
    )

    agg = (
        df.groupby(["display_model_name", "data_source", "tgt_lang", layer_col], dropna=False)[value_col]
        .mean()
        .reset_index()
        if not df.empty
        else df
    )

    figsize = figsize or (max(13, 2.3 * len(target_data_langs)), max(5.1, 2.55 * len(models)))
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        len(models),
        len(target_data_langs),
        hspace=0.24,
        wspace=grid_wspace,
    )
    axes: list[list[tuple[Any, Any, Any]]] = []
    for row_idx in range(len(models)):
        row_axes = []
        for col_idx in range(len(target_data_langs)):
            inner = outer[row_idx, col_idx].subgridspec(
                3,
                1,
                height_ratios=[0.36, 0.36, 1.85],
                hspace=0.04,
            )
            row_axes.append(
                (
                    fig.add_subplot(inner[0, 0]),
                    fig.add_subplot(inner[1, 0]),
                    fig.add_subplot(inner[2, 0]),
                )
            )
        axes.append(row_axes)

    plotted_langs = set()
    for row_idx, model in enumerate(models):
        row_layers = _coerce_plot_layers(df[df["display_model_name"] == model], layer_col)
        row_x_min, row_x_max = _layer_axis_limits(row_layers)
        for col_idx, target_data_lang in enumerate(target_data_langs):
            data_source = _target_data_source_for_lang(target_data_lang)
            strip_lang_ax, strip_vocab_ax, line_ax = axes[row_idx][col_idx]

            entropy_df = _aggregate_entropy(
                prompt_df,
                data_source=data_source,
                display_model_name=model,
                prompt_langs=prompt_langs,
                layer_col=layer_col,
            )
            _plot_entropy_strip(
                strip_lang_ax,
                entropy_df,
                layer_values=row_layers,
                layer_col=layer_col,
                value_col="entropy",
                label="Lang Entrp",
                cmap="viridis",
                vmin=entropy_limits["entropy"][0],
                vmax=entropy_limits["entropy"][1],
                show_label=col_idx == 0,
            )
            _plot_entropy_strip(
                strip_vocab_ax,
                entropy_df,
                layer_values=row_layers,
                layer_col=layer_col,
                value_col="tgt_first_token_vocab_entropy_bits",
                label="Vocab Entrp",
                cmap="magma",
                vmin=entropy_limits["tgt_first_token_vocab_entropy_bits"][0],
                vmax=entropy_limits["tgt_first_token_vocab_entropy_bits"][1],
                show_label=col_idx == 0,
            )

            panel = agg[
                (agg["display_model_name"] == model)
                & (agg["data_source"] == data_source)
            ]
            plotted_langs.update(
                _line_panel(
                    line_ax,
                    panel,
                    layer_col=layer_col,
                    value_col=value_col,
                    target_langs=target_langs,
                    colors=colors,
                    bold_lang=target_data_lang,
                    yscale=yscale,
                    ylim=ylim,
                )
            )
            if row_idx == 0:
                strip_lang_ax.set_title(
                    _format_lang_label(target_data_lang),
                    fontsize=14,
                    fontfamily="Avenir Next",
                    fontweight=600,
                    pad=6,
                )
            if col_idx == 0:
                _add_model_row_label(line_ax, model)
            line_ax.set_xlim(row_x_min, row_x_max)
            if layer_col == "layer":
                line_ax.xaxis.set_major_locator(MultipleLocator(10))
            strip_lang_ax.set_xlim(row_x_min, row_x_max)
            strip_vocab_ax.set_xlim(row_x_min, row_x_max)
            line_ax.tick_params(axis="x", labelbottom=True, pad=1)
            line_ax.tick_params(axis="y", left=col_idx == 0, labelleft=col_idx == 0)

    _set_split_figure_title(fig, title, title_fontsize=16, title_fontweight=600, subtitle_fontsize=12)
    grid_top = 0.865 if title else 0.92
    fig.supylabel(ylabel, fontsize=15, x=ylabel_x, y=(_CONTROLLED_STARTW_GRID_BOTTOM + grid_top) / 2)

    handles = [
        plt.Line2D([0], [0], color=colors[lang], lw=2.0, label=_format_lang_label(lang))
        for lang in target_langs
        if lang in plotted_langs
    ]
    style_handles = [
        plt.Line2D([0], [0], color="#222222", lw=2.6, linestyle="-", label="latent lang == translation target"),
        plt.Line2D([0], [0], color="#222222", lw=2.0, linestyle="--", label="top-3 peak langs if not column lang"),
    ]
    _add_controlled_startw_bottom_layout(
        fig,
        language_handles=handles,
        style_handles=style_handles,
        layer_label="Layer" if layer_col == "layer" else _axis_label_for_layer_col(layer_col),
        top=grid_top,
        left=grid_left,
        right=grid_right,
    )
    _add_entropy_colorbars(
        fig,
        entropy_limits=entropy_limits,
        right=grid_right,
        lang_top=axes[0][0][2].get_position().y1,
        vocab_bottom=axes[-1][0][2].get_position().y0,
    )
    return fig


def plot_startw_translation_source_grid(
    menu_df: pd.DataFrame,
    prompt_df: pd.DataFrame,
    *,
    models: Sequence[str],
    prompt_langs: Sequence[str],
    target_data_langs: Sequence[str] | None = None,
    target_langs: Sequence[str] | None = None,
    layer_col: str = "layer",
    value_col: str = "menu_mean_prob_share",
    yscale: str = "log",
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ylabel: str = "Start(w) probability per latent language",
):
    """Static translation grid with columns as source prompt languages.

    Curves are menu target languages. Values are averaged over the requested
    translation-to-target data sources within each source-language panel.
    """
    if target_data_langs is None:
        data_sources = sorted(
            source
            for source in menu_df["data_source"].dropna().unique()
            if isinstance(source, str) and source.startswith("translation_to_")
        )
    else:
        data_sources = [_target_data_source_for_lang(lang) for lang in target_data_langs]
    df = menu_df[
        (menu_df["display_model_name"].isin(list(models)))
        & (menu_df["data_source"].isin(data_sources))
        & (menu_df["prompt_lang"].isin(list(prompt_langs)))
    ].copy()
    if target_langs is not None:
        df = df[df["tgt_lang"].isin(list(target_langs))]
    prompt_langs = _ordered_present(df["prompt_lang"].unique(), prompt_langs)
    target_langs = _ordered_present(df["tgt_lang"].unique(), target_langs)
    colors = _target_string_lang_colors(target_langs)
    entropy_limits = _entropy_limits(
        prompt_df,
        data_sources=data_sources,
        models=models,
        prompt_langs=prompt_langs,
        layer_col=layer_col,
    )

    agg = (
        df.groupby(["display_model_name", "prompt_lang", "tgt_lang", layer_col], dropna=False)[value_col]
        .mean()
        .reset_index()
        if not df.empty
        else df
    )

    figsize = figsize or (max(13, 2.3 * len(prompt_langs)), max(5.1, 2.55 * len(models)))
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        len(models),
        len(prompt_langs),
        hspace=0.24,
        wspace=0.08,
    )
    axes: list[list[tuple[Any, Any, Any]]] = []
    for row_idx in range(len(models)):
        row_axes = []
        for col_idx in range(len(prompt_langs)):
            inner = outer[row_idx, col_idx].subgridspec(
                3,
                1,
                height_ratios=[0.36, 0.36, 1.85],
                hspace=0.04,
            )
            row_axes.append(
                (
                    fig.add_subplot(inner[0, 0]),
                    fig.add_subplot(inner[1, 0]),
                    fig.add_subplot(inner[2, 0]),
                )
            )
        axes.append(row_axes)

    plotted_langs = set()
    for row_idx, model in enumerate(models):
        row_layers = _coerce_plot_layers(df[df["display_model_name"] == model], layer_col)
        row_x_min, row_x_max = _layer_axis_limits(row_layers)
        for col_idx, prompt_lang in enumerate(prompt_langs):
            strip_lang_ax, strip_vocab_ax, line_ax = axes[row_idx][col_idx]

            entropy_df = prompt_df[
                (prompt_df["data_source"].isin(data_sources))
                & (prompt_df["display_model_name"] == model)
                & (prompt_df["prompt_lang"] == prompt_lang)
            ].copy()
            entropy_cols = [layer_col]
            for col in ["entropy", "tgt_first_token_vocab_entropy_bits"]:
                if col in entropy_df.columns:
                    entropy_cols.append(col)
            if len(entropy_cols) > 1 and not entropy_df.empty:
                entropy_df = entropy_df[entropy_cols].groupby(layer_col, as_index=False).mean(numeric_only=True)
            else:
                entropy_df = pd.DataFrame(columns=[layer_col, "entropy", "tgt_first_token_vocab_entropy_bits"])
            _plot_entropy_strip(
                strip_lang_ax,
                entropy_df,
                layer_values=row_layers,
                layer_col=layer_col,
                value_col="entropy",
                label="Lang Entrp",
                cmap="viridis",
                vmin=entropy_limits["entropy"][0],
                vmax=entropy_limits["entropy"][1],
                show_label=col_idx == 0,
            )
            _plot_entropy_strip(
                strip_vocab_ax,
                entropy_df,
                layer_values=row_layers,
                layer_col=layer_col,
                value_col="tgt_first_token_vocab_entropy_bits",
                label="Vocab Entrp",
                cmap="magma",
                vmin=entropy_limits["tgt_first_token_vocab_entropy_bits"][0],
                vmax=entropy_limits["tgt_first_token_vocab_entropy_bits"][1],
                show_label=col_idx == 0,
            )

            panel = agg[
                (agg["display_model_name"] == model)
                & (agg["prompt_lang"] == prompt_lang)
            ]
            plotted_langs.update(
                _line_panel(
                    line_ax,
                    panel,
                    layer_col=layer_col,
                    value_col=value_col,
                    target_langs=target_langs,
                    colors=colors,
                    bold_lang=prompt_lang,
                    yscale=yscale,
                    ylim=ylim,
                )
            )
            if row_idx == 0:
                strip_lang_ax.set_title(
                    _format_lang_label(prompt_lang),
                    fontsize=14,
                    fontfamily="Avenir Next",
                    fontweight=600,
                    pad=6,
                )
            if col_idx == 0:
                _add_model_row_label(line_ax, model)
            line_ax.set_xlim(row_x_min, row_x_max)
            if layer_col == "layer":
                line_ax.xaxis.set_major_locator(MultipleLocator(10))
            strip_lang_ax.set_xlim(row_x_min, row_x_max)
            strip_vocab_ax.set_xlim(row_x_min, row_x_max)
            line_ax.tick_params(axis="x", labelbottom=True, pad=1)
            line_ax.tick_params(axis="y", left=col_idx == 0, labelleft=col_idx == 0)

    _set_split_figure_title(fig, title, title_fontsize=16, title_fontweight=600, subtitle_fontsize=12)
    grid_top = 0.865 if title else 0.92
    fig.supylabel(ylabel, fontsize=15, x=0.075, y=(_CONTROLLED_STARTW_GRID_BOTTOM + grid_top) / 2)

    handles = [
        plt.Line2D([0], [0], color=colors[lang], lw=2.0, label=_format_lang_label(lang))
        for lang in target_langs
        if lang in plotted_langs
    ]
    style_handles = [
        plt.Line2D([0], [0], color="#222222", lw=2.6, linestyle="-", label="latent lang == translation source"),
        plt.Line2D([0], [0], color="#222222", lw=2.0, linestyle="--", label="top-3 peak langs if not column lang"),
    ]
    _add_controlled_startw_bottom_layout(
        fig,
        language_handles=handles,
        style_handles=style_handles,
        layer_label="Layer" if layer_col == "layer" else _axis_label_for_layer_col(layer_col),
        top=grid_top,
    )
    _add_entropy_colorbars(
        fig,
        entropy_limits=entropy_limits,
        right=0.90,
        lang_top=axes[0][0][2].get_position().y1,
        vocab_bottom=axes[-1][0][2].get_position().y0,
    )
    return fig


################################################################################
# Open-ended and model-comparison styling
################################################################################

OPENENDED_METHOD_COLORS = {
    "repr": "#4C78A8",
    "raw-rmax": "#F58518",
    "raw-rtopp": "#54A24B",
    "tuned-rmax": "#B279A2",
    "tuned-rtopp": "#9D755D",
}

DOMAIN_COLORS = {
    "pud9": "#FDBA74",
    "pud9_ud6": "#FB923C",
    "pud21": "#F97316",
    "pud21_ud6": "#C2410C",
    "arts_humanities": "#A78BFA",
    "social_science": "#6366F1",
    "social_sciences": "#6366F1",
    "stem": "#1D4ED8",
}

PUD_DOMAIN_ORDER = [
    "pud9",
    "pud9_ud6",
    "pud21",
    "pud21_ud6",
]

INCLUDE_DOMAIN_ORDER = [
    "arts_humanities",
    "social_science",
    "stem",
]

DOMAIN_ORDER = PUD_DOMAIN_ORDER + INCLUDE_DOMAIN_ORDER

DOMAIN_LABELS = {
    "arts_humanities": "Arts/Humanities",
    "social_science": "Social Sciences",
    "social_sciences": "Social Sciences",
    "stem": "STEM",
    "pud9": "PUD9",
    "pud9_ud6": "PUD9 + UD6",
    "pud21": "PUD21",
    "pud21_ud6": "PUD21 + UD6",
}

LLID_METRIC_LABELS = {
    "pivot": "Pivot Rate",
    "entropy": "Entropy",
    "dominance": "Dominance",
    "agreement_repr_vs_raw-rmax": "Agreement: Repr vs Raw Argmax",
    "agreement_repr_vs_raw-rtopp": "Agreement: Repr vs Raw Top-p",
}


def _llid_metric_label(metric: str) -> str:
    return LLID_METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _metric_display_name(metric: str) -> str:
    return {
        "dominance": "Dominance",
        "pivot": "Pivot Rate",
    }.get(metric, metric.replace("_", " ").title())


def _domain_label(domain: Any, *, prefix_include: bool = False) -> str:
    label = DOMAIN_LABELS.get(domain, str(domain).replace("_", " ").title())
    if prefix_include and domain in set(INCLUDE_DOMAIN_ORDER + ["social_sciences"]):
        return f"INCLUDE: {label}"
    return label


def _domain_linestyle(domain: Any) -> str:
    return "--" if domain in set(INCLUDE_DOMAIN_ORDER + ["social_sciences"]) else "-"


STORY_PROB_CATEGORY_ORDER = ["task_relevant", "english", "other"]
STORY_PROB_CATEGORY_LABELS = {
    "task_relevant": "P(L=task-relevant)",
    "english": "P(L=English)",
    "other": "P(L=other)",
}
STORY_PROB_CATEGORY_COLORS = {
    "task_relevant": "#2f7ed8",
    "english": "#d95f02",
    "other": "#6a6a6a",
}


################################################################################
# Story figures
################################################################################


def plot_story_category_bar_grid(
    summary_df: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    models: Sequence[str],
    row_order: Sequence[str] | None = None,
    col_order: Sequence[str] | None = None,
    category_order: Sequence[str] = tuple(STORY_PROB_CATEGORY_ORDER),
    category_labels: Mapping[str, str] = STORY_PROB_CATEGORY_LABELS,
    category_colors: Mapping[str, str] = STORY_PROB_CATEGORY_COLORS,
    probability_label: str = "Average probability",
    title: str | None = None,
    subtitle: str | None = None,
    unavailable_label: str = "results",
    figsize: tuple[float, float] | None = None,
    show_row_label_in_ylabel: bool = True,
    legend_y: float = 0.015,
    bottom: float = 0.23,
    top: float | None = None,
    left: float | None = None,
    right: float | None = None,
    title_y: float = 0.995,
    subtitle_y: float = 0.955,
    shared_ylabel: str | None = None,
    shared_ylabel_x: float = 0.012,
    model_label_map: Mapping[str, str] | None = None,
    sharey: bool = True,
    row_label_fontweight: str | int = "normal",
    row_label_fontsize: float = 13,
    col_title_fontweight: str | int = "normal",
    ylabel_pad: float = 4.0,
    x_tick_label_pad: float | None = None,
    subtitle_linespacing: float = 1.0,
):
    """Plot story bars in a row/column facet grid.

    Each panel has models on the x-axis and grouped language-category bars.
    The y-axis is shared and scaled from the global max including error bars.
    """
    if summary_df.empty:
        print(f"No rows available for {unavailable_label}.")
        return None
    needed = {row_col, col_col, "display_model_name", "prob_category", "mean_prob"}
    missing = sorted(needed - set(summary_df.columns))
    if missing:
        raise ValueError(f"Missing columns for story bar grid: {missing}")

    df = summary_df.copy()
    rows = list(row_order or _ordered_present(df[row_col]))
    cols = list(col_order or _ordered_present(df[col_col]))
    models = [model for model in models if model in set(df["display_model_name"])]
    if not rows or not cols or not models:
        print(f"No matching rows/columns/models available for {unavailable_label}.")
        return None

    finite_values = pd.to_numeric(df["mean_prob"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    error_col = "stderr_prob" if "stderr_prob" in df.columns else "std_prob"
    if error_col in df.columns:
        finite_values = finite_values + pd.to_numeric(df[error_col], errors="coerce").fillna(0.0)
    global_max = float(finite_values.max()) if finite_values.notna().any() else 1.0
    ylim_top = max(1e-6, min(1.0, global_max * 1.18 if np.isfinite(global_max) else 1.0))

    figsize = figsize or (
        max(11.0, 4.6 * len(cols)),
        max(3.8, 2.9 * len(rows)),
    )
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=sharey,
        gridspec_kw={"hspace": 0.24, "wspace": 0.08},
    )
    width = 0.22 if len(category_order) <= 3 else 0.72 / max(1, len(category_order))
    x = np.arange(len(models))

    for row_idx, row_value in enumerate(rows):
        for col_idx, col_value in enumerate(cols):
            ax = axes[row_idx, col_idx]
            panel = df[(df[row_col] == row_value) & (df[col_col] == col_value)]
            pivot = (
                panel.pivot_table(
                    index="display_model_name",
                    columns="prob_category",
                    values="mean_prob",
                    aggfunc="mean",
                )
                .reindex(index=models, columns=list(category_order))
            )
            err = None
            if error_col in panel.columns:
                err = (
                    panel.pivot_table(
                        index="display_model_name",
                        columns="prob_category",
                        values=error_col,
                        aggfunc="mean",
                    )
                    .reindex(index=models, columns=list(category_order))
                )
            if panel.empty or not np.isfinite(pivot.to_numpy(dtype=float)).any():
                ax.text(
                    0.5,
                    0.5,
                    f"{unavailable_label} not available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="#666666",
                )
            for cat_idx, category in enumerate(category_order):
                offset = (cat_idx - (len(category_order) - 1) / 2.0) * width
                values = pivot[category].to_numpy(dtype=float)
                yerr = err[category].to_numpy(dtype=float) if err is not None and category in err else None
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    color=category_colors.get(category),
                    label=category_labels.get(category, category),
                    yerr=yerr,
                    error_kw={"elinewidth": 0.7, "capsize": 2.0, "alpha": 0.7},
                )
            ax.set_ylim(0.0, ylim_top)
            ax.grid(axis="y", linewidth=0.4, alpha=0.25)
            if row_idx == 0:
                ax.set_title(
                    str(col_value),
                    fontsize=15,
                    pad=8,
                    fontweight=col_title_fontweight,
                )
            if col_idx == 0:
                if shared_ylabel:
                    ylabel = str(row_value) if show_row_label_in_ylabel else ""
                else:
                    ylabel = f"{row_value}\n{probability_label}" if show_row_label_in_ylabel else probability_label
                ax.set_ylabel(
                    ylabel,
                    fontsize=row_label_fontsize,
                    fontweight=row_label_fontweight,
                    labelpad=ylabel_pad,
                )
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
            ax.tick_params(labelsize=12)
            for tick_label in ax.get_yticklabels():
                tick_label.set_fontweight("bold")

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        model_tick_labels = [model_label_map.get(model, model) for model in models] if model_label_map else models
        ax.set_xticklabels(model_tick_labels, rotation=35, ha="right")
        if x_tick_label_pad is not None:
            ax.tick_params(axis="x", pad=x_tick_label_pad)
    for ax in axes[:-1, :].ravel():
        ax.tick_params(labelbottom=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(handles),
            frameon=False,
            fontsize=14,
        )
    if title:
        if subtitle and "\n" not in subtitle:
            subtitle_parts = subtitle.split("; ", 2)
            if len(subtitle_parts) == 3:
                subtitle = f"{subtitle_parts[0]}; {subtitle_parts[1]};\n{subtitle_parts[2]}"
        full_title = title if not subtitle else f"{title}\n{subtitle}"
        _set_split_figure_title(
            fig,
            full_title,
            title_fontsize=18,
            subtitle_fontsize=14,
            subtitle_linespacing=subtitle_linespacing,
            title_y=title_y,
            subtitle_y=subtitle_y,
        )
    if shared_ylabel:
        fig.supylabel(shared_ylabel, x=shared_ylabel_x, fontsize=15, fontweight="normal")
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top if top is not None else (0.84 if title else 0.95),
    )
    return fig


def overlay_story_category_max_markers(
    fig,
    average_summary_df: pd.DataFrame,
    max_summary_df: pd.DataFrame,
    *,
    row_col: str,
    col_col: str,
    row_order: Sequence[str],
    col_order: Sequence[str],
    models: Sequence[str],
    category_order: Sequence[str] = tuple(STORY_PROB_CATEGORY_ORDER),
    category_colors: Mapping[str, str] = STORY_PROB_CATEGORY_COLORS,
    max_categories: Sequence[str] = ("task_relevant", "other"),
    max_legend_label: str = "Avg. Max. P(L)",
    legend_y: float = 0.015,
    legend_fontsize: float = 18,
    x_edge_padding: float = 0.46,
    x_tick_shift_points: float = 14.0,
):
    """Overlay category maxima and shared styling on a story bar grid.

    Bars in ``average_summary_df`` retain their own SE whiskers. For each
    requested category, this adds a connector and hollow diamond at the mean
    per-row category maximum, with SE taken from ``max_summary_df``.
    """
    if fig is None or average_summary_df.empty or max_summary_df.empty:
        return fig

    rows = list(row_order)
    cols = list(col_order)
    models = list(models)
    panel_count = len(rows) * len(cols)
    if panel_count == 0 or len(fig.axes) < panel_count:
        return fig
    panel_axes = np.asarray(fig.axes[:panel_count], dtype=object).reshape(len(rows), len(cols))

    width = 0.22 if len(category_order) <= 3 else 0.72 / max(1, len(category_order))
    category_offsets = {
        category: (idx - (len(category_order) - 1) / 2.0) * width
        for idx, category in enumerate(category_order)
    }
    max_error_col = "stderr_prob" if "stderr_prob" in max_summary_df.columns else "std_prob"

    for row_idx, row_value in enumerate(rows):
        for col_idx, col_value in enumerate(cols):
            ax = panel_axes[row_idx, col_idx]
            for model_idx, model in enumerate(models):
                for category in max_categories:
                    if category not in category_offsets:
                        continue
                    selector = (
                        average_summary_df[row_col].eq(row_value)
                        & average_summary_df[col_col].eq(col_value)
                        & average_summary_df["display_model_name"].eq(model)
                        & average_summary_df["prob_category"].eq(category)
                    )
                    max_selector = (
                        max_summary_df[row_col].eq(row_value)
                        & max_summary_df[col_col].eq(col_value)
                        & max_summary_df["display_model_name"].eq(model)
                        & max_summary_df["prob_category"].eq(category)
                    )
                    average_rows = average_summary_df[selector]
                    max_rows = max_summary_df[max_selector]
                    if average_rows.empty or max_rows.empty:
                        continue
                    average_prob = float(average_rows["mean_prob"].mean())
                    max_prob = float(max_rows["mean_prob"].mean())
                    max_se = (
                        float(max_rows[max_error_col].mean())
                        if max_error_col in max_rows.columns
                        else np.nan
                    )
                    x_pos = model_idx + category_offsets[category]
                    color = category_colors.get(category, "#333333")
                    ax.vlines(x_pos, average_prob, max_prob, color=color, linewidth=1.5, zorder=4)
                    ax.errorbar(
                        x_pos,
                        max_prob,
                        yerr=max_se if np.isfinite(max_se) else None,
                        fmt="D",
                        markersize=5.5,
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markeredgewidth=1.5,
                        ecolor=color,
                        elinewidth=1.2,
                        capsize=3.0,
                        zorder=6,
                    )
            ax.set_xlim(-x_edge_padding, len(models) - 1 + x_edge_padding)

    scale_values = []
    for summary in (average_summary_df, max_summary_df):
        values = pd.to_numeric(summary["mean_prob"], errors="coerce")
        error_col = "stderr_prob" if "stderr_prob" in summary.columns else "std_prob"
        if error_col in summary.columns:
            values = values + pd.to_numeric(summary[error_col], errors="coerce").fillna(0.0)
        scale_values.append(values)
    finite_values = pd.concat(scale_values, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if not finite_values.empty:
        ylim_top = max(1e-6, min(1.0, float(finite_values.max()) * 1.18))
        for ax in panel_axes.ravel():
            ax.set_ylim(0.0, ylim_top)

    shift = plt.matplotlib.transforms.ScaledTranslation(
        x_tick_shift_points / 72.0,
        0.0,
        fig.dpi_scale_trans,
    )
    for ax in panel_axes.ravel():
        for tick_label in ax.get_xticklabels():
            if tick_label.get_visible() and tick_label.get_text():
                tick_label.set_transform(tick_label.get_transform() + shift)

    handles, labels = panel_axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D(
        [],
        [],
        linestyle="none",
        marker="D",
        markersize=6.0,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.5,
    ))
    labels.append(max_legend_label)
    for old_legend in list(fig.legends):
        old_legend.remove()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(handles),
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=1.35,
        handletextpad=0.55,
    )

    if getattr(fig, "_supylabel", None) is not None:
        panel_y0 = min(ax.get_position().y0 for ax in panel_axes.ravel())
        panel_y1 = max(ax.get_position().y1 for ax in panel_axes.ravel())
        fig._supylabel.set_y((panel_y0 + panel_y1) / 2.0)
    return fig


################################################################################
# PUD and INCLUDE figures
################################################################################


def plot_domain_comparison_lines(
    prompt_df: pd.DataFrame,
    *,
    metric: str,
    data_source: str | None = None,
    data_sources: Sequence[str] | None = None,
    methods: Sequence[str] = ("repr", "raw-rmax", "raw-rtopp"),
    models: Sequence[str] = ("Llama-2-7B", "Aya-23-8B", "Apertus-8B"),
    domains: Sequence[str] | None = None,
    category_col: str = "domain_category",
    layer_col: str = "layer",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    panel_label_fontsize: float = 11,
    panel_label_fontweight: str | int = "normal",
    tick_fontsize: float = 8,
    legend_fontsize: float = 9,
    axis_label_fontsize: float = 13,
    x_axis_label_y: float = 0.120,
    y_axis_label_y: float = 0.5,
):
    if data_sources is None:
        data_sources = [data_source or "include_10lang_3domain_cap30"]
    elif data_source is not None:
        data_sources = list(data_sources) + [data_source]
    data_sources = list(dict.fromkeys(data_sources))
    category_col = category_col if category_col in prompt_df.columns else "include_domain"
    df = prompt_df[
        prompt_df["data_source"].isin(data_sources)
        & prompt_df["method_label"].isin(list(methods))
        & prompt_df["display_model_name"].isin(list(models))
    ].copy()
    domains = list(domains or _ordered_present(df[category_col], DOMAIN_ORDER))
    agg = (
        df.groupby(["display_model_name", "method_label", category_col, layer_col], dropna=False)[metric]
        .mean()
        .reset_index()
        if not df.empty
        else df
    )
    figsize = figsize or (max(9, 3.0 * len(methods)), max(5.5, 1.9 * len(models)))
    fig, axes = plt.subplots(
        len(models),
        len(methods),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.14},
    )
    for row_idx, model in enumerate(models):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx, col_idx]
            for domain in domains:
                series = agg[
                    (agg["display_model_name"] == model)
                    & (agg["method_label"] == method)
                    & (agg[category_col] == domain)
                ].sort_values(layer_col)
                if series.empty:
                    continue
                ax.plot(
                    series[layer_col],
                    series[metric],
                    color=DOMAIN_COLORS.get(domain),
                    linestyle=_domain_linestyle(domain),
                    linewidth=1.8,
                    label=domain,
                )
            ax.set_ylim(0, 1)
            ax.grid(True, linewidth=0.4, alpha=0.25)
            if row_idx == 0:
                ax.set_title(
                    method,
                    fontsize=panel_label_fontsize,
                    fontweight=panel_label_fontweight,
                )
            if col_idx == 0:
                ax.set_ylabel(
                    model,
                    fontsize=panel_label_fontsize,
                    fontweight=panel_label_fontweight,
                    labelpad=10,
                )
            ax.tick_params(labelsize=tick_fontsize)
    pud_domains = [domain for domain in domains if domain in PUD_DOMAIN_ORDER]
    include_domains = [domain for domain in domains if domain not in set(pud_domains)]
    pud_label_handle = plt.Line2D([0], [0], color="none", lw=0, label="PUD:")
    include_label_handle = plt.Line2D([0], [0], color="none", lw=0, label="INCLUDE:")
    pud_handles = [
        plt.Line2D(
            [0],
            [0],
            color=DOMAIN_COLORS.get(domain),
            linestyle=_domain_linestyle(domain),
            lw=2.2,
            label=_domain_label(domain),
        )
        for domain in pud_domains
    ]
    include_handles = [
        plt.Line2D(
            [0],
            [0],
            color=DOMAIN_COLORS.get(domain),
            linestyle=_domain_linestyle(domain),
            lw=2.2,
            label=_domain_label(domain),
        )
        for domain in include_domains
    ]
    if pud_handles and include_handles:
        pud_legend = fig.legend(
            handles=[pud_label_handle] + pud_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.060),
            ncol=len(pud_handles) + 1,
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=2.2,
            columnspacing=1.2,
        )
        include_legend = fig.legend(
            handles=[include_label_handle] + include_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=len(include_handles) + 1,
            frameon=False,
            fontsize=legend_fontsize,
            handlelength=2.2,
            columnspacing=1.2,
        )
        for legend in (pud_legend, include_legend):
            if legend.get_texts():
                legend.get_texts()[0].set_fontweight("bold")
        fig.subplots_adjust(left=0.16, bottom=0.20, top=0.84 if title else 0.94)
    else:
        handles = pud_handles + include_handles
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            fontsize=legend_fontsize,
        )
        fig.subplots_adjust(left=0.16, bottom=0.14, top=0.84 if title else 0.94)
    _set_split_figure_title(fig, title)
    fig.supxlabel(
        "Layer" if layer_col == "layer" else _axis_label_for_layer_col(layer_col),
        fontsize=axis_label_fontsize,
        y=x_axis_label_y,
    )
    fig.supylabel(
        _metric_display_name(metric),
        fontsize=axis_label_fontsize,
        x=0.060,
        y=y_axis_label_y,
    )
    return fig


################################################################################
# Base-instruct and training-dynamics figures
################################################################################


def plot_base_vs_instruct_method_grid(
    summary_df: pd.DataFrame,
    *,
    data_source: str,
    families: Sequence[str],
    layer_col: str = "layer",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Compare base/instruct trajectories for repr, raw top-p, and agreement."""
    set_matplotlib_paper_font()
    df = summary_df[
        summary_df["data_source"].eq(data_source)
        & summary_df["base_instruct_family"].isin(list(families))
    ].copy()
    metric_groups = (
        (
            "Pivot rate",
            (
                ("pivot", "repr", "Repr-GMM"),
                ("pivot", "raw-rtopp", "Raw LogitLens Top-p"),
            ),
        ),
        (
            "Entropy",
            (
                ("entropy", "repr", "Repr-GMM"),
                ("entropy", "raw-rtopp", "Raw LogitLens Top-p"),
            ),
        ),
        (
            "Agreement",
            (("agreement", "agreement_repr_vs_raw-rtopp", "Repr-Decoding"),),
        ),
    )
    panel_specs = tuple(
        panel_spec
        for _, group_panels in metric_groups
        for panel_spec in group_panels
    )
    variants = ("base", "instruct")
    variant_colors = {"base": "#0072B2", "instruct": "#E69F00"}
    variant_linestyles = {"base": "-", "instruct": "--"}
    figsize = figsize or (18.0, max(7.5, 2.35 * len(families)))
    fig = plt.figure(figsize=figsize)
    outer_grid = fig.add_gridspec(
        1,
        len(metric_groups),
        width_ratios=[len(group_panels) for _, group_panels in metric_groups],
        wspace=0.17,
    )
    axes = np.empty((len(families), len(panel_specs)), dtype=object)
    group_column_ranges = []
    first_x_axis = None
    column_offset = 0
    for group_idx, (_, group_panels) in enumerate(metric_groups):
        group_grid = outer_grid[0, group_idx].subgridspec(
            len(families),
            len(group_panels),
            hspace=0.30,
            wspace=0.045 if len(group_panels) > 1 else 0.0,
        )
        group_columns = range(column_offset, column_offset + len(group_panels))
        group_column_ranges.append(tuple(group_columns))
        for row_idx in range(len(families)):
            row_group_axis = None
            for local_col_idx, col_idx in enumerate(group_columns):
                ax = fig.add_subplot(
                    group_grid[row_idx, local_col_idx],
                    sharex=first_x_axis,
                    sharey=row_group_axis,
                )
                axes[row_idx, col_idx] = ax
                if first_x_axis is None:
                    first_x_axis = ax
                if row_group_axis is None:
                    row_group_axis = ax
        column_offset += len(group_panels)

    for row_idx, family in enumerate(families):
        for col_idx, (metric, method_label, method_label_display) in enumerate(panel_specs):
            ax = axes[row_idx, col_idx]
            for variant in variants:
                series = df[
                    df["base_instruct_family"].eq(family)
                    & df["base_instruct_variant"].eq(variant)
                    & df["metric"].eq(metric)
                    & df["method_label"].eq(method_label)
                ].sort_values(layer_col)
                if series.empty:
                    continue
                ax.plot(
                    series[layer_col],
                    series["value"],
                    color=variant_colors[variant],
                    linestyle=variant_linestyles[variant],
                    linewidth=2.8,
                    label=variant.title(),
                )
            if metric in {"pivot", "agreement"}:
                ax.set_ylim(0.0, 1.0)
            else:
                ax.set_ylim(bottom=0.0)
            ax.grid(True, linewidth=0.45, alpha=0.25)
            ax.tick_params(labelsize=14)
            if row_idx == 0:
                ax.set_title(method_label_display, fontsize=16, fontweight=600, pad=12)
            if col_idx == 0:
                ax.set_ylabel(family, fontsize=17, fontweight=600, labelpad=12)
            elif col_idx not in {2, 4}:
                ax.tick_params(axis="y", labelleft=False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=variant_colors[variant],
            linestyle=variant_linestyles[variant],
            linewidth=3.2,
            label=variant.title(),
        )
        for variant in variants
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.018),
        ncol=2,
        frameon=False,
        fontsize=19,
        handlelength=2.6,
        columnspacing=1.8,
    )
    if title:
        fig.suptitle(title, fontsize=19, fontweight=600, y=0.985)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.18, top=0.80 if title else 0.84)

    for (metric_label, _), group_columns in zip(metric_groups, group_column_ranges):
        left_pos = axes[0, group_columns[0]].get_position()
        right_pos = axes[0, group_columns[-1]].get_position()
        group_midpoint = (left_pos.x0 + right_pos.x1) / 2.0
        rule_y = left_pos.y1 + 0.055
        fig.add_artist(
            plt.Line2D(
                [left_pos.x0, right_pos.x1],
                [rule_y, rule_y],
                transform=fig.transFigure,
                color="#333333",
                linewidth=1.4,
                solid_capstyle="butt",
            )
        )
        fig.text(
            group_midpoint,
            rule_y + 0.018,
            metric_label,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight=600,
        )
        fig.text(
            group_midpoint,
            0.120,
            "Layer",
            ha="center",
            va="center",
            fontsize=18,
        )
    return fig


def _format_training_tokens(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    value = float(value)
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2g}T"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.0f}B"
    return f"{value:.0f}"


def plot_training_dynamics_layer_grid(
    summary_df: pd.DataFrame,
    *,
    model_families: Sequence[str],
    metrics: Sequence[str] = ("pivot", "dominance", "entropy", "agreement_repr_vs_raw-rmax"),
    layer_col: str = "layer",
    value_col: str = "value",
    family_col: str = "training_model_family",
    token_col: str = "training_tokens",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot checkpoint trajectories across layers, colored by training tokens."""
    set_matplotlib_paper_font()
    df = summary_df[
        summary_df[family_col].isin(list(model_families))
        & summary_df["metric"].isin(list(metrics))
    ].copy()
    if df.empty:
        figsize = figsize or (max(10, 3.1 * len(metrics)), max(4.8, 1.7 * len(model_families)))
        fig, axes = plt.subplots(
            len(model_families),
            len(metrics),
            figsize=figsize,
            squeeze=False,
            sharex=True,
            gridspec_kw={"hspace": 0.32, "wspace": 0.24},
        )
        for row_idx, family in enumerate(model_families):
            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                if row_idx == 0:
                    ax.set_title(_llid_metric_label(metric), fontsize=11, pad=6)
                if col_idx == 0:
                    ax.set_ylabel(family, fontsize=10, labelpad=10)
                ax.text(
                    0.5,
                    0.5,
                    "no runs found",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#737373",
                )
                ax.grid(True, linewidth=0.4, alpha=0.25)
        _set_split_figure_title(fig, title)
        fig.supxlabel("Layer", fontsize=13, y=0.055)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.17, top=0.84 if title else 0.94)
        return fig

    figsize = figsize or (max(10, 3.1 * len(metrics)), max(5.0, 1.8 * len(model_families)))
    fig, axes = plt.subplots(
        len(model_families),
        len(metrics),
        figsize=figsize,
        squeeze=False,
        sharex=True,
        gridspec_kw={"hspace": 0.32, "wspace": 0.24},
    )

    token_values = pd.to_numeric(df[token_col], errors="coerce") if token_col in df.columns else pd.Series(dtype=float)
    finite_tokens = token_values[np.isfinite(token_values)]
    cmap = plt.colormaps.get_cmap("viridis")
    if finite_tokens.empty or finite_tokens.min() == finite_tokens.max():
        norm = None
    else:
        norm = plt.Normalize(vmin=float(finite_tokens.min()), vmax=float(finite_tokens.max()))

    def token_color(token_value: Any):
        token_float = pd.to_numeric(pd.Series([token_value]), errors="coerce").iloc[0]
        if norm is None or pd.isna(token_float):
            return "#2563EB"
        return cmap(norm(float(token_float)))

    for row_idx, family in enumerate(model_families):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            panel = df[(df[family_col] == family) & (df["metric"] == metric)].copy()
            if panel.empty:
                ax.text(
                    0.5,
                    0.5,
                    "missing",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#737373",
                )
            else:
                panel["_token_sort"] = pd.to_numeric(panel[token_col], errors="coerce")
                for token_value, series in panel.sort_values(["_token_sort", layer_col]).groupby(token_col, dropna=False):
                    series = series.sort_values(layer_col)
                    ax.plot(
                        series[layer_col],
                        series[value_col],
                        color=token_color(token_value),
                        linewidth=1.8,
                        alpha=0.90,
                    )
            if metric in {"pivot", "dominance"} or metric.startswith("agreement_"):
                ax.set_ylim(0.0, 1.0)
            elif metric == "entropy":
                ax.set_ylim(bottom=0.0)
            ax.grid(True, linewidth=0.4, alpha=0.25)
            if row_idx == 0:
                ax.set_title(_llid_metric_label(metric), fontsize=11, pad=6)
            if col_idx == 0:
                ax.set_ylabel(family, fontsize=10, labelpad=10)
            ax.tick_params(labelsize=8)

    _set_split_figure_title(fig, title)
    fig.supxlabel("Layer", fontsize=13, y=0.055)
    fig.subplots_adjust(left=0.13, right=0.90, bottom=0.17, top=0.84 if title else 0.94)

    if norm is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
        cb.set_label("Training tokens", fontsize=9)
        ticks = np.linspace(float(finite_tokens.min()), float(finite_tokens.max()), num=min(5, finite_tokens.nunique()))
        cb.set_ticks(ticks)
        cb.set_ticklabels([_format_training_tokens(tick) for tick in ticks])
        cb.ax.tick_params(labelsize=8)
    return fig
