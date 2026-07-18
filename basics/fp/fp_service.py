from datetime import datetime
from typing import List, Tuple
from datetime import datetime
from typing import Protocol

from datetime import datetime
from fp_data import *


class ActiveFlightRepository(Protocol):
    """Infrastructure interface to query active or pending flights."""
    
    def get_flights_intersecting_area(
        self, 
        polygon: List[Tuple[float, float]], 
        start: datetime, 
        end: datetime
    ) -> List[FlightPlan]:
        """Queries spatial database (e.g., PostGIS) for routes overlapping the event zone."""
        ...

    def update_flight_status(self, plan_id: str, is_approved: bool, errors: List[str]) -> None:
        ...

class AirspaceRegistry(Protocol):
    """Protocol for fetching real-time, dynamic airspace rules."""

    def get_active_restrictions(
        self,
        bounding_area: list[Waypoint],
        start: datetime,
        end: datetime,
    ) -> list[AirspaceRestriction]:
        """Fetch restrictions overlapping the given space and time."""
        ...

class RouteValidator(Protocol):
    """Protocol for processing geometric and temporal intersection logic."""

    def validate(
        self, route: FlightRoute, restrictions: list[AirspaceRestriction]
    ) -> ValidationResult:
        """Validate a route against a set of active restrictions."""
        ...

MAX_RETRIES_PLAN_ROUTE = 10

class NotificationService(Protocol):
    def send(self, message: str) -> None: ...
