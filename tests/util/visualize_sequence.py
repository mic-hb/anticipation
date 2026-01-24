from typing import Optional, Any, Dict, Iterable, List
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from tests.util.entities import (
    Event,
    EventSpecialCode,
    get_midi_instrument_name_from_midi_instrument_code,
)


def _events_to_df(events: Iterable[Event]) -> pd.DataFrame:
    # Materialize so Iterable/generator behavior is deterministic.
    events_list = list(events)
    if len(events_list) == 0:
        raise ValueError("events is empty")

    norm: List[tuple[Any, ...]] = []
    for i, e in enumerate(events_list):
        # idx here is "sequence index order" (input order).
        norm.append(
            (
                i,
                e.midi_time(),
                e.midi_duration(),
                e.midi_instrument(),
                e.midi_note(),
                e.note().to_name(),
                e.special_code,
                e.is_control,
            )
        )

    df = pd.DataFrame(
        norm,
        columns=[
            "idx",
            "start",
            "duration",
            "program",
            "note",
            "note_name",
            "special_code",
            "is_control",
        ],
    )
    if (df["duration"] < 0).any():
        raise ValueError("Found negative durations.")
    df["end"] = df["start"] + df["duration"]
    return df


def _build_program_palette(programs: list[int]) -> Dict[int, str]:
    palette = (
        qualitative.Plotly
        + qualitative.D3
        + qualitative.G10
        + qualitative.T10
        + qualitative.Dark24
        + qualitative.Light24
    )
    programs_sorted = sorted(programs)
    return {p: palette[i % len(palette)] for i, p in enumerate(programs_sorted)}


def _add_boundaries_to_subplot(fig: go.Figure, df_boundaries: pd.DataFrame) -> None:
    """
    Adds boundary lines + annotations to the *top* subplot only.
    Uses subplot-aware add_shape/add_annotation so it doesn't span the index subplot.
    """
    dfb = df_boundaries.sort_values(by="start", kind="mergesort")
    for row in dfb.itertuples(index=False):
        t = int(row.start)
        special_code = row.special_code

        if special_code == EventSpecialCode.AUTOREGRESSIVE_TOKEN:
            display_text = "autoregress"
        elif special_code == EventSpecialCode.ANTICIPATION_TOKEN:
            display_text = "anticipate"
        elif special_code == EventSpecialCode.SEQ_SEPARATION_TOKENS:
            display_text = "new sample"
        else:
            display_text = "?"

        if special_code == EventSpecialCode.SEQ_SEPARATION_TOKENS:
            line_props = {"width": 1}
            yshift = 0
        else:
            line_props = {"dash": "dash", "width": 1}
            yshift = 10

        # Boundary line (top subplot only)
        fig.add_shape(
            type="line",
            x0=t,
            x1=t,
            y0=0,
            y1=1,
            xref="x",
            yref="y domain",
            line=line_props,
            layer="above",
        )

        # Annotation (top subplot only)
        fig.add_annotation(
            x=t,
            y=1,
            xref="x",
            yref="y domain",
            text=display_text,
            showarrow=False,
            yanchor="bottom",
            yshift=yshift,
        )


def plot_pianoroll_with_index_timeline(
    events: list[Event],
    *,
    title: str = "MIDI Piano Roll",
    height: int = 1000,
    keep_rangeslider: bool = True,
    y_jitter: float = 0.18,
    x_jitter: int = 1,
) -> go.Figure:
    df = _events_to_df(events)
    is_boundary = df["special_code"] != EventSpecialCode.TYPICAL_EVENT
    df_boundaries = df.loc[is_boundary].copy()
    df_notes = df.loc[~is_boundary].copy()

    if df_notes.empty:
        raise ValueError(
            "No note events remain after filtering special-code boundary events."
        )

    programs_sorted = sorted(df_notes["program"].unique().tolist())
    program_to_color = _build_program_palette(programs_sorted)

    # Subplots: top = roll, bottom = index timeline
    fig = go.Figure()

    # --- Row 1: Piano roll (one trace per program) ---
    hover_template_roll = (
        "Program: %{meta.program} (%{meta.program_name})<br>"
        "Note: %{customdata[0]} (%{customdata[1]})<br>"
        "Start: %{customdata[2]} ticks<br>"
        "End: %{customdata[3]} ticks<br>"
        "Dur: %{customdata[4]} ticks<br>"
        "Idx: %{customdata[5]}"
        "<extra></extra>"
    )

    # Small vertical offsets to separate overlapping notes
    # magnitude chosen to be visually clear but musically negligible
    program_offsets = {
        p: (
            (i - (len(programs_sorted) - 1) / 2) * y_jitter,
            (i - (len(programs_sorted) - 1) / 2) * x_jitter,
        )
        for i, p in enumerate(programs_sorted)
    }

    for midi_program_code in programs_sorted:
        sub = df_notes[df_notes["program"] == midi_program_code].sort_values(
            ["start", "note"], kind="mergesort"
        )
        midi_program_name = get_midi_instrument_name_from_midi_instrument_code(
            midi_program_code
        )

        # jitter so we can distinguish between different instruments that
        # happen at the exact same time on the exact same note
        y_off, x_off = program_offsets[midi_program_code]

        for label, sub2, dash, width, opacity in [
            ("notes", sub[sub["is_control"] == False], "solid", 8, 1.0),
            ("control", sub[sub["is_control"] == True], "1, 1, 1", 8, 1.0),
        ]:
            if sub2.empty:
                continue

            x: list[Optional[float]] = []
            y: list[Optional[float]] = []
            custom: list[Optional[list[Any]]] = []

            for idx, start, end, note, note_name, dur in sub2[
                ["idx", "start", "end", "note", "note_name", "duration"]
            ].itertuples(index=False):
                x0 = int(start) + x_off
                x1 = int(end) + x_off
                yv = int(note) + y_off

                x.extend([x0, x1, None])
                y.extend([yv, yv, None])

                row = [note_name, int(note), int(start), int(end), int(dur), int(idx)]
                custom.extend([row, row, None])

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    customdata=custom,
                    mode="lines+markers",
                    line=dict(
                        # width=6, color=program_to_color[midi_program_code]
                        width=6,
                        dash=dash,
                        color=program_to_color[midi_program_code],
                    ),
                    marker=dict(
                        symbol="line-ns",
                        size=11,
                        angleref="previous",
                        standoff=3,
                    ),
                    opacity=0.72,
                    name=f"{midi_program_code}: {midi_program_name}",
                    meta={
                        "program": int(midi_program_code),
                        "program_name": midi_program_name,
                    },
                    showlegend=(True if label == "notes" else False),
                    hovertemplate=hover_template_roll,
                    connectgaps=False,
                    legendgroup=f"prog_{midi_program_code}",
                ),
            )

    df_seq = df_notes.sort_values(["idx"], kind="mergesort").copy()
    df_seq["x_mid"] = (df_seq["start"] + df_seq["end"]) / 2.0

    fig.update_layout(
        xaxis=dict(
            title="ticks from start",
            type="linear",
            rangeslider=dict(visible=keep_rangeslider, thickness=0.08),
        ),
        xaxis2=dict(
            overlaying="x",  # draw in same plot area
            matches="x",  # keep identical range/zoom
            visible=False,  # hide axis decorations
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=df_seq["x_mid"].to_list(),
            y=df_seq["note"].to_list(),
            mode="lines+markers+text",
            name="Sequence order path",
            visible="legendonly",  # starts off; user can toggle on in legend
            line=dict(width=1),  # keep subtle; increase if desired
            marker=dict(
                symbol="arrow",
                size=11,
                angleref="previous",
                standoff=3,
            ),
            text=df_seq["idx"].astype(int).astype(str).to_list(),
            textposition="top center",
            textfont=dict(size=10),
            hovertemplate=(
                "Idx: %{text}<br>Time(mid): %{x:.0f} ticks<br>Note: %{y}<extra></extra>"
            ),
            showlegend=True,
            xaxis="x2",  # <-- key line
        ),
    )
    _add_boundaries_to_subplot(fig, df_boundaries)
    fig.update_yaxes(title_text="MIDI note")
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=70, r=30, t=60, b=60),
        hovermode="closest",
        dragmode=False,  # preserves hover-first interaction
        legend_title_text="Program",
    )

    return fig


def write_fig_html(
    fig: go.Figure, save_to_path: Path, *, auto_open: bool = True
) -> None:
    assert save_to_path.suffix == ".html"
    fig.write_html(
        str(save_to_path.absolute()),
        auto_open=auto_open,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
        },
    )


def get_figure_and_open(
    events: list[Event],
    path: Path,
    auto_open: bool = True,
) -> None:
    fig1 = plot_pianoroll_with_index_timeline(events)
    write_fig_html(fig1, path, auto_open=auto_open)
