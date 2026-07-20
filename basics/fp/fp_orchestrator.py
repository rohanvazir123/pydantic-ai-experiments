import uuid

from fp_data import *
from fp_service import *

class FlightPlanner:
    def __init__(self, registry: AirspaceRegistry, validator: RouteValidator, notification_service : NotificationService):
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
        relevant_restrictions : list[AirspaceRestriction] = self._registry.get_active_restrictions(
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


