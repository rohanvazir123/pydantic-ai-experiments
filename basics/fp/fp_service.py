from abc import ABC, abstractmethod

class AirspaceRegistry(ABC):
    """Interface for fetching real-time, dynamic airspace rules."""
    
    @abstractmethod
    def get_active_restrictions(
        self, 
        bounding_area: List[Waypoint], 
        start: datetime, 
        end: datetime
    ) -> List[AirspaceRestriction]:
        pass

class RouteValidator(ABC):
    """Interface for processing the geometric and temporal intersection logic."""
    
    @abstractmethod
    def validate(
        self, 
        route: FlightRoute, 
        restrictions: List[AirspaceRestriction]
    ) -> ValidationResult:
        pass

from abc import ABC, abstractmethod

class ActiveFlightRepository(ABC):
    """Infrastructure interface to query active or pending flights."""
    
    @abstractmethod
    def get_flights_intersecting_area(
        self, 
        polygon: List[Tuple[float, float]], 
        start: datetime, 
        end: datetime
    ) -> List[FlightPlan]:
        """Queries spatial database (e.g., PostGIS) for routes overlapping the event zone."""
        pass

    @abstractmethod
    def update_flight_status(self, plan_id: str, is_approved: bool, errors: List[str]) -> None:
        pass
