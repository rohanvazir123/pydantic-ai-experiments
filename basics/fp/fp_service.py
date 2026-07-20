from abc import ABC, abstractmethod

from datetime import datetime
from fp_data import *


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
