'''Task heads and losses for beat/chord learning.'''

from .beat import BeatConfig, BeatHead, BeatLoss
from .chord import ChordConfig, ChordHead, ChordLoss

__all__ = [
    'BeatConfig',
    'BeatHead',
    'BeatLoss',
    'ChordConfig',
    'ChordHead',
    'ChordLoss',
]
