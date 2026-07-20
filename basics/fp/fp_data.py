from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


@dataclass(frozen=True, kw_only=True)
class Waypoint:
    latitude: float
    longitude: float
    altitude_ft: float
    eta: datetime

@dataclass(frozen=True, kw_only=True)
class FlightRoute:
    waypoints: list[Waypoint]
    aircraft_type: str

@dataclass(frozen=True, kw_only=True)
class AltitudeRange:
    floor_ft: float
    ceiling_ft: float

class RestrictionType(StrEnum):
    PROHIBITED = "prohibited"
    RESTRICTED = "restricted"
    DANGER = "danger"
    WEATHER_RESTRICTED = "weather_restricted"
    TFR = "tfr"  # Temporary Flight Restriction

@dataclass(frozen=True, kw_only=True)
class AirspaceRestriction:
    restriction_id: str
    type: RestrictionType
    boundary_polygon: list[tuple[float, float]]  # list of (lat, lon) coordinates
    altitudes: AltitudeRange
    start_time: datetime
    end_time: datetime

@dataclass(frozen=True, kw_only=True)
class ValidationResult:
    is_valid: bool
    violations: list[str]

@dataclass(frozen=True, kw_only=True)
class AirspaceUpdatedEvent:
    """Triggered when a new restriction is published or modified."""
    restriction_id: str
    affected_polygon: list[tuple[float, float]]
    floor_ft: float
    ceiling_ft: float
    start_time: datetime
    end_time: datetime

@dataclass(frozen=True, kw_only=True)
class Waypoint:
    latitude: float
    longitude: float
    altitude_ft: float
    eta: datetime


@dataclass(frozen=True, kw_only=True)
class FlightRoute:
    waypoints: list[Waypoint]
    aircraft_type: str

@dataclass(frozen=True, kw_only=True)
class AltitudeRange:
    floor_ft: float
    ceiling_ft: float

@dataclass(frozen=True, kw_only=True)
class ValidationResult:
    is_valid: bool
    violations: list[str]

@dataclass(frozen=True, kw_only=True)
class RouteRequest:
    flight_id: str
    proposed_waypoints: list[Waypoint]
    aircraft_type: str

@dataclass(frozen=True, kw_only=True)
class FlightPlan:
    plan_id: str
    flight_id: str
    route: FlightRoute
    alternate_routes: list[FlightRoute]
    is_approved: bool
    validation_errors: list[str]

