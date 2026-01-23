from typing import Tuple, Optional, Any, Dict, Iterable
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from tests.util.entities import (
    Note,
    Event,
    EventSpecialCode,
    get_midi_instrument_name_from_midi_instrument_code,
)


def _plot_midi_ticks_by_program(
    events: Iterable[Event],
    *,
    title: str = "MIDI Piano Roll",
    show_note_names_on_y: bool = True,
    height: int = 650,
    max_y_ticks: int = 40,
    x_range: Optional[Tuple[int, int]] = None,
    keep_rangeslider: bool = True,
) -> go.Figure:
    """
    events: (start_tick, duration_ticks, midi_note[, program])
            If program is omitted, defaults to 0.

    Rendering strategy:
      - one trace per program => stable per-program color
      - within each trace, many disconnected line segments via None separators => good performance
      - drag interactions disabled on main plot (hover friendly), rangeslider retained
    """
    if not events:
        raise ValueError("events is empty")

    # Normalize input tuples to 4 columns.
    norm = []
    for e in events:
        e: Event
        p_tuple = (
            e.midi_time(),
            e.midi_duration(),
            e.midi_instrument(),
            e.midi_note(),
            e.note().to_name(),
            e.special_code,
        )
        norm.append(p_tuple)

    df = pd.DataFrame(
        norm,
        columns=["start", "duration", "program", "note", "note_name", "special_code"],
    )

    if (df["duration"] < 0).any():
        raise ValueError("Found negative durations.")
    df["end"] = df["start"] + df["duration"]

    is_boundary = df["special_code"] != EventSpecialCode.TYPICAL_EVENT
    df_boundaries = df.loc[is_boundary].copy()
    df_notes = df.loc[~is_boundary].copy()

    # Choose a deterministic color palette.
    palette = (
        qualitative.Plotly
        + qualitative.D3
        + qualitative.G10
        + qualitative.T10
        + qualitative.Dark24
        + qualitative.Light24
    )

    programs_sorted = sorted(df_notes["program"].unique().tolist())
    program_to_color: Dict[int, str] = {
        p: palette[i % len(palette)] for i, p in enumerate(programs_sorted)
    }
    fig = go.Figure()
    hover_template = (
        "Program: %{meta.program} (%{meta.program_name})<br>"
        "Note: %{customdata[0]} (%{customdata[1]})<br>"
        "Start: %{customdata[2]} ticks<br>"
        "End: %{customdata[3]} ticks<br>"
        "Dur: %{customdata[4]} ticks"
        "<extra></extra>"
    )

    for midi_program_code in programs_sorted:
        sub = df_notes[df_notes["program"] == midi_program_code].sort_values(
            ["start", "note"], kind="mergesort"
        )
        midi_program_name = get_midi_instrument_name_from_midi_instrument_code(
            midi_program_code
        )

        x: list[Optional[int]] = []
        y: list[Optional[int]] = []
        custom: list[Optional[list[Any]]] = []

        for start, end, note, note_name, dur in sub[
            ["start", "end", "note", "note_name", "duration"]
        ].itertuples(index=False):
            x.extend([int(start), int(end), None])
            y.extend([int(note), int(note), None])
            row = [
                note_name,
                int(note),
                int(start),
                int(end),
                int(dur),
            ]
            custom.extend([row, row, None])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                customdata=custom,
                mode="lines",
                line=dict(width=6, color=program_to_color[midi_program_code]),
                name=f"{midi_program_code}: {midi_program_name}",
                meta={
                    "program": int(midi_program_code),
                    "program_name": midi_program_name,
                },
                hovertemplate=hover_template,
                connectgaps=False,
            )
        )

    # Add vertical dashed boundary lines that span the whole plotting area.
    boundary_ticks = df_boundaries.sort_values(by="start")
    shapes = []
    annotations = []
    for boundary_tick in boundary_ticks.itertuples(index=False):
        t = boundary_tick.start
        special_code = boundary_tick.special_code
        if special_code == EventSpecialCode.AUTOREGRESSIVE_TOKEN:
            display_text = "autoregress"
        elif special_code == EventSpecialCode.ANTICIPATION_TOKEN:
            display_text = "anticipate"
        elif special_code == EventSpecialCode.SEQ_SEPARATION_TOKENS:
            display_text = "new sample"
        else:
            display_text = "?"

        if special_code == EventSpecialCode.SEQ_SEPARATION_TOKENS:
            # new samples are visually separated by a non-dashed line
            line_props = {"width": 1}
            # this yshfit stuff is so that both these lines can be
            # at the same time and their annotation text does not overlap
            yshift = 0
        else:
            line_props = {"dash": "dash", "width": 1}
            yshift = 10

        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=t,
                x1=t,
                y0=0,
                y1=1,
                line=line_props,
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=t,
                y=1,
                xref="x",
                yref="paper",
                text=display_text,
                showarrow=False,
                yanchor="bottom",
                yshift=yshift,
            )
        )

    if shapes and annotations:
        fig.update_layout(shapes=shapes, annotations=annotations)

    # Axes / UX
    fig.update_xaxes(
        title_text="ticks from start",
        type="linear",
        rangeslider=dict(visible=keep_rangeslider),
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))

    fig.update_yaxes(title_text="MIDI note")

    if show_note_names_on_y:
        note_min = int(df_notes["note"].min())
        note_max = int(df_notes["note"].max())
        span = max(1, note_max - note_min)
        step = max(1, span // max_y_ticks)
        tick_vals = list(range(note_min, note_max + 1, step))
        tick_text = [Note.make(n).to_name() for n in tick_vals]
        fig.update_yaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text)

    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=70, r=30, t=60, b=60),
        hovermode="closest",
        dragmode=False,
        legend_title_text="Program",
    )

    return fig


def get_figure_and_open(
    events: list[Event], save_to_path: Path, auto_open: bool = True
) -> None:
    """Creates an interactive visualization of a list of events.
    Args:
        events: list of Event objects, parsed from token sequence
        save_to_path: place on disk to save the visualization, it is an
            html file created by plotly. It should end with the file
            type suffix ".html"
        auto_open: if true, opens in the default app for html files.
            Otherwise, just saves file to disk. True by default.

    Returns:
        nothing
    """
    assert save_to_path.suffix == ".html"
    fig = _plot_midi_ticks_by_program(events)
    fig.write_html(
        str(save_to_path.absolute()),
        auto_open=auto_open,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
        },
    )
