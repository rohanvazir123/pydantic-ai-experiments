from typing import Protocol
from dataclasses import dataclass


@dataclass
class Route:
    origin: str
    dest: str
    fuel_kg: float
    storm_risk: float


class Validator(Protocol):
    def __call__(self, route: Route) -> bool: ...


def validate_weather(route: Route) -> bool:
    return route.storm_risk < 0.3


def validate_fuel(route: Route) -> bool:
    return route.fuel_kg >= 5000


VALIDATORS: list[Validator] = [validate_weather, validate_fuel]


def is_cleared(route: Route) -> bool:
    return all(v(route) for v in VALIDATORS)


print(is_cleared(Route("SFO", "JFK", 6000, 0.1)))   # True
print(is_cleared(Route("SFO", "JFK", 3000, 0.1)))   # False