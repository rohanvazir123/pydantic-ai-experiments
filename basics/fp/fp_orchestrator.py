'''
You said: Flight Planner API: 
Design the class structure and clean API interfaces 
for a flight planning system that validates routes based on airspace restrictions.
'''

'''
You have identified the core operational components: an orchestrator, 
a data structure for the route, a repository for the rules, 
and a validation engine.

Where your design can improve is by decoupling 
your data models from your business logic and 
introducing proper abstractions for external infrastructure. 

In a clean API design, 
AerialRoute and AirspaceRules should be passive data models (or configurations), 
while the validator and planner handle the execution.
Here is a breakdown of how to refine your classes, 
followed by a clean, production-ready API design.

Structural Refinements
Separate Data from Logic: 
Keep FlightRoute (your AerialRoute) 
purely as a data structure containing waypoints, altitudes, and times.

Abstract the Rule Sources: Airspace restrictions change constantly. 
Instead of a static AirspaceRules class, 
use an AirspaceRegistry or AirspaceProvider interface that can fetch dynamic restrictions based on time and geometry.
Inversion of Control: The FlightPlanner shouldn't tightly couple to a single validator. 
Inject a validator interface so you can easily swap or chain different rules (e.g., weather validation, fuel validation).

classDiagram
    class FlightPlanner {
        -IAirspaceRegistry registry
        -IRouteValidator validator
        +planRoute(RouteRequest request) FlightPlan
    }
    class IAirspaceRegistry {
        <<interface>>
        +getRestrictions(Geometry area, DateTime time) list~AirspaceRestriction~
    }
    class IRouteValidator {
        <<interface>>
        +validate(FlightRoute route, list~AirspaceRestriction~ restrictions) ValidationResult
    }
    class FlightRoute {
        +list~Waypoint~ waypoints
        +String aircraftType
    }
    class AirspaceRestriction {
        +String id
        +RestrictionType type
        +Geometry boundary
        +AltitudeRange altitudeLimits
    }
    
    FlightPlanner --> IAirspaceRegistry
    FlightPlanner --> IRouteValidator
    IRouteValidator ..> FlightRoute
    IRouteValidator ..> AirspaceRestriction


'''

import uuid

from fp_data import *

from abc import ABC, abstractmethod

class FlightPlanner:
    def __init__(self, registry: AirspaceRegistry, 
                 validator: RouteValidator, notification_service : NotificationService):
        self._registry = registry
        self._validator = validator
        self._notification_service = notification_service # Interface to alert pilots/dispatch

    def plan_route(self, request: RouteRequest) -> FlightPlan:
        max_retries, retries = MAX_RETRIES_PLAN_ROUTE, 0

        # 1. Initialize the immutable domain route object
        route = FlightRoute(
            waypoints=request.proposed_waypoints, 
            aircraft_type=request.aircraft_type
        )
        
        # 2. Extract timelines for query efficiency
        start_time = route.waypoints[0].eta
        end_time = route.waypoints[-1].eta
        
        # 3. Fetch applicable geo/temporal restrictions from infrastructure
        relevant_restrictions = self._registry.get_active_restrictions(
            bounding_area=route.waypoints, 
            start=start_time, 
            end=end_time
        )
        
        # 4. Delegate business rules calculation to the pure validation engine
        validation = self._validator.validate(route, relevant_restrictions)

        while not validation.is_valid and retries < max_retries:
            # If the exercise requires mitigation, delegate to the rerouter
            alternative_route, relevant_restrictions = self._rerouter.suggest_alternative(route, validation.conflicting_restrictions)
            # Re-validate the new suggestion
            validation = self._validator.validate(alternative_route, relevant_restrictions)

            # backoff and retry
            retries += 1

        if validation.is_valid:
            route = alternative_route
            alternative_route = None

        
        # 5. Compile and return finalized plan receipt
        return FlightPlan(
            plan_id=str(uuid.uuid4()),
            route=route,
            is_approved=validation.is_valid,
            validation_errors=validation.violations,
            alternative_route =alternative_route
        )
    
    def handle_airspace_update(self, event: AirspaceUpdatedEvent) -> None:
        """Asynchronous event handler invoked by a message broker (e.g., Kafka/RabbitMQ)."""
        
        # 1. Map event data into our domain Restriction model
        new_restriction = AirspaceRestriction(
            restriction_id=event.restriction_id,
            type=RestrictionType.TFR,
            boundary_polygon=event.affected_polygon,
            altitudes=AltitudeRange(floor_ft=event.floor_ft, ceiling_ft=event.ceiling_ft),
            start_time=event.start_time,
            end_time=event.end_time
        )

        # 2. Query DB to see which active flight trajectories intersect this 4D space
        potentially_affected_plans = self._repository.get_flights_intersecting_area(
            polygon=event.affected_polygon,
            start=event.start_time,
            end=event.end_time
        )

        # 3. Precisely re-evaluate each flight
        for plan in potentially_affected_plans:
            validation = self._validator.validate(plan.route, [new_restriction])
            
            if not validation.is_valid:
                # 4. Revoke approval and notify stakeholders immediately
                self._repository.update_flight_status(
                    plan_id=plan.plan_id, 
                    is_approved=False, 
                    errors=validation.violations
                )
                
                self._notification_service.dispatch_conflict_alert(
                    plan_id=plan.plan_id,
                    message=f"CRITICAL: Flight route compromised by new TFR {event.restriction_id}."
                )



class RouteRerouter(ABC):
    """Responsible for calculating alternate trajectories if validation fails."""
    
    @abstractmethod
    def suggest_alternative(
        self, 
        original_route: FlightRoute, 
        violations: list[AirspaceRestriction]
    ) -> FlightRoute:
        pass


class NotificationService(Protocol):
    def send(self, message: str) -> None: ...
