"""Shared transformations for low-level MIDI events."""

from __future__ import annotations

from collections.abc import Iterable

import mido


def remove_control_changes(
    track: mido.MidiTrack,
    *,
    control_numbers: Iterable[int],
) -> mido.MidiTrack:
    """Remove selected controls while preserving every other event's tick."""

    removed_controls = {int(control) for control in control_numbers}
    rebuilt_track = mido.MidiTrack()
    carried_delta = 0
    for message in track:
        if (
            message.type == "control_change"
            and int(message.control) in removed_controls
        ):
            carried_delta += int(message.time)
            continue
        rebuilt_track.append(message.copy(time=int(message.time) + carried_delta))
        carried_delta = 0
    if carried_delta:
        rebuilt_track.append(mido.MetaMessage("end_of_track", time=carried_delta))
    return rebuilt_track


__all__ = ("remove_control_changes",)
