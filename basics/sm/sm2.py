from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Generic, TypeVar

# Define the explicit type variables for clarity if needed by older tools,
# but PEP 695 handles S, E, C directly within the class scope.
@dataclass
class StateMachine2[S: StrEnum, E: StrEnum, C]:
    current_state: S
    context: C
    # Transition table mapping: (CurrentState, Event) -> NextState
    transitions: dict[tuple[S, E], S] = field(default_factory=dict)
    # Callback actions: Event -> Function(Context)
    actions: dict[E, Callable[[C], None]] = field(default_factory=dict)

    def add_transition(self, from_state: S, event: E, to_state: S) -> None:
        """Registers a valid state change."""
        self.transitions[(from_state, event)] = to_state

    def add_action(self, event: E, action: Callable[[C], None]) -> None:
        """Registers a side effect to execute when an event occurs."""
        self.actions[event] = action

    def send_event(self, event: E) -> None:
        """Triggers a transition and fires accompanying actions."""
        lookup = (self.current_state, event)
        
        if lookup not in self.transitions:
            raise ValueError(f"Invalid transition from {self.current_state} via {event}")
            
        # Execute side effect first
        if event in self.actions:
            self.actions[event](self.context)
            
        # Transition the state
        self.current_state = self.transitions[lookup]



# 1. Define concrete Enums and Context data
class AuthState(StrEnum):
    LOGGED_OUT = "logged_out"
    LOGGING_IN = "logging_in"
    LOGGED_IN = "logged_in"

class AuthEvent(StrEnum):
    SUBMIT = "submit"
    SUCCESS = "success"
    LOGOUT = "logout"

@dataclass
class AuthContext:
    username: str
    token: str | None = None

# 2. Instantiate with explicit type binding
context = AuthContext(username="Alice")
auth_fsm = StateMachine2[AuthState, AuthEvent, AuthContext](
    current_state=AuthState.LOGGED_OUT, 
    context=context
)

# 3. Configure the state transitions
auth_fsm.add_transition(AuthState.LOGGED_OUT, AuthEvent.SUBMIT, AuthState.LOGGING_IN)
auth_fsm.add_transition(AuthState.LOGGING_IN, AuthEvent.SUCCESS, AuthState.LOGGED_IN)
auth_fsm.add_transition(AuthState.LOGGED_IN, AuthEvent.LOGOUT, AuthState.LOGGED_OUT)

# 4. Attach a type-safe side effect
def handle_success(ctx: AuthContext) -> None:
    ctx.token = "secret_jwt_token"
    print(f"Token assigned to {ctx.username}!")

auth_fsm.add_action(AuthEvent.SUCCESS, handle_success)

# 5. Run the machine
auth_fsm.send_event(AuthEvent.SUBMIT)
auth_fsm.send_event(AuthEvent.SUCCESS)  # Prints: Token assigned to Alice!

print(auth_fsm.current_state)  # Output: AuthState.LOGGED_IN
print(auth_fsm.context.token)   # Output: secret_jwt_token
