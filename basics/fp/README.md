# FLIGHT ROUTE PLANNER

## BASIC IMPORTS

```
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List
```

## DATA MODELS
```
@dataclass(frozen=True)
class Waypoint:
float_lat: float; float_lon: float; alt_ft: float; eta: datetime

@dataclass(frozen=True)
class FlightRoute:
waypoints: List[Waypoint]; aircraft_type: str

@dataclass(frozen=True)
class FlightPlan:
plan_id: str; route: FlightRoute; is_approved: bool; errors: List[str]
```

## CORE SYSTEM INTERFACES
```
class RouteValidator(ABC):
    """Stateless engine for 4D intersection business logic."""
    @abstractmethod
    def validate(self, route: FlightRoute, rules: List[dict]) -> dict: pass

class FlightPlanner(ABC):
    """Synchronous orchestrator for initial flight lifecycle management."""
    @abstractmethod
    def plan_route(self, route_request: FlightRoute) -> FlightPlan: pass
```

## EVENT / SUBSCRIPTION INTERFACES
```
class AirspaceSubscriptionService(ABC):
    """Handles real-time ingestion of airspace changes and subscriber push updates."""
    @abstractmethod
    def subscribe_to_sector(self, sector_id: str, plan_id: str) -> None: pass
    
    @abstractmethod
    def broadcast_route_modification(self, plan_id: str, new_conflicts: List[str]) -> None: pass
```
