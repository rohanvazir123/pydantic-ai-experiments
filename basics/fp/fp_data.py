from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, auto
from typing import List, Tuple

class RestrictionType(StrEnum):
    PROHIBITED = "prohibited"
    RESTRICTED = "restricted"
    DANGER = "danger"
    WEATHER_RESTRICTED = "weather_restricted"
    TFR = "tfr"  # Temporary Flight Restriction

@dataclass(frozen=True, kw_only=True)
class Waypoint:
    latitude: float
    longitude: float
    altitude_ft: float
    eta: datetime

@dataclass(frozen=True, kw_only=True)
class FlightRoute:
    waypoints: List[Waypoint]
    aircraft_type: str

@dataclass(frozen=True, kw_only=True)
class AltitudeRange:
    floor_ft: float
    ceiling_ft: float

@dataclass(frozen=True, kw_only=True)
class AirspaceRestriction:
    restriction_id: str
    type: RestrictionType
    boundary_polygon: List[Tuple[float, float]]  # List of (lat, lon) coordinates
    altitudes: AltitudeRange
    start_time: datetime
    end_time: datetime

@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    violations: List[str]


from dataclasses import dataclass
from datetime import datetime

@dataclass(frozen=True)
class AirspaceUpdatedEvent:
    """Triggered when a new restriction is published or modified."""
    restriction_id: str
    affected_polygon: List[Tuple[float, float]]
    floor_ft: float
    ceiling_ft: float
    start_time: datetime
    end_time: datetime

