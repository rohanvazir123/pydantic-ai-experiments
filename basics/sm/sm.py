from dataclasses import dataclass, field
from typing import Callable, Protocol
from enum import Enum, StrEnum
from itertools import Iterable

import uuid

type Action[C] = Callable[[C], None]
type GuardRail[C] = Callable[[C], None]
type Transition[S, E, C] = dict[ tuple[S, E], tuple[S, Action[C]], GuardRail[C] ]

class InvalidStateTransitionError(Exception):
    pass

class GuardRailValidationError(Exception):
    pass

@dataclass
class StateMachine[S: StrEnum, E: StrEnum, C]:
    transitions: Transition[S, E, C] = field(default_factory=Transition[S, E, C])
    current_state: S = None

    def add_transition(self, from_state: S, event: E, to_state: S, \
            action: Action[C] = None, guard_rails: list[GuardRail[C]] = []) -> None:
        self.transitions[from_state][event] = (to_state, guard_rails, action)

    def set_initial_state(self, state):
        # Set the initial state of the state machine
        if state in self.states:
            self.current_state = state
        else:
            raise ValueError(f"State {state} is not a valid state.")
        
    def _next_transition(self, event: E) -> tuple[S, Action[C], list[GuardRail]]:
        # Get the next state and action based on the current state and event
        try:
            if not self.current_state:
                raise InvalidStateTransitionError("Current state is not set. " \
                "Cannot perform transition.")
            return self.transitions[self.current_state][event]

        except KeyError:
            raise InvalidStateTransitionError(f"No transition defined for state \
                {self.current_state} on event {event}.")
        
    def handle_event(self, ctx: C, event: E) -> None:
   
        try:
             # Get the next transition
            new_state, action, guard_rails = self._next_transition(event)

            # Check for all guard rails
            all(guard_rail(ctx) for guard_rail in guard_rails if isinstance(guard_rail, GuardRail))

            # Take action
            if action and isinstance(action, Action[C]):
                action(ctx)

            # Set current state to new state ONLY after all clear
            self.current_state = new_state
        except InvalidStateTransitionError as e:
            print(f"Error: {e}")
            raise
        except Exception as e:
            print(f"Exception : {e}")
            raise

        return self.current_state

def basic_test():
    class PaymentState(StrEnum):
        PROCESSING = "PROCESSING"
        AUTHORIZED = "Authorized"
        CAPTURED = "Captured"
        PROCESSING = "Processing"
        COMPLETED = "Completed"
        FAILED = "Failed"
        CANCELLED = "Cancelled"
        REFUNDED = "Refunded"

    class PaymentEvent(StrEnum):
        AUTHORIZE = "authorize"
        CAPTURE = "capture"
        REFUND = "refund"
        CANCEL = "cancel"
        COMPLETE = "complete"
        FAIL = "failed"

    @dataclass(kw_only=True)
    class PaymentCtx:
        user_id : str
        account_id : str
        account_balance: float
        txn_amount: float
        txn_id  = field(init=False) 
        audit : list[str] = field(default_factory=list)

        def __post_init__(self):
            self.txn_id = str(uuid.uuid4())
        
        def print_audit_log(self):
            print("Audit Log:")
            for entry in self.audit:
                print(entry)

    class InvalidPaymentAccountBalance(Exception):
        pass   


    class InvalidPaymentTransaction(Exception):
        pass

    class InvalidPaymentAccount(Exception):
        pass   


    class PaymentAction:

        def authorize(self, ctx: PaymentCtx) -> None:
            ctx.audit.append(f"{ctx.txn_id}: Payment authorized")

        def capture(self, ctx: PaymentCtx) -> None:
            ctx.audit.append(f"{ctx.txn_id}: Payment captured")

        def fail(self, ctx: PaymentCtx) -> None:
            ctx.audit.append(f"{ctx.txn_id}: Payment failed")

        def refund(self, ctx: PaymentCtx) -> None:
            ctx.audit.append(f"{ctx.txn_id}: Payment refunded")

        def cancel(self, ctx: PaymentCtx) -> None:    
            ctx.audit.append(f"{ctx.txn_id}: Payment cancelled")

        def complete(self, ctx: PaymentCtx) -> None:
            ctx.audit.append(f"{ctx.txn_id}: Payment completed")

    class PaymentValidator(Protocol):
        def __call__(self, ctx: PaymentCtx) -> bool: ...

    # Payment validators
    def is_account_valid(ctx: PaymentCtx) -> None:
        # Use ctx to determine is account is valid
        if not ctx.account_valid:
            raise InvalidPaymentAccount

    def is_account_balance_valid(ctx: PaymentCtx) -> None:
        # Use ctx to determine account balance valid
        if  ctx.account_balance <= 1000:
            raise InvalidPaymentAccountBalance
    
    def is_payment_transaction_valid(ctx: PaymentCtx) -> None:
        # Use ctx to check txn validity
        if ctx.txn_amount <= 0 or ctx.txn_amount > 10000:
            raise InvalidPaymentTransaction

    PAYMENT_VALIDATORS: dict[str, PaymentValidator] = {
        "account": is_account_valid,
        "balance": is_account_balance_valid,
        "transaction": is_payment_transaction_valid,
    }

    def payment_validators_for(names: Iterable[str]) -> list[PaymentValidator]:
        return [PAYMENT_VALIDATORS[n] for n in names]

    # Create a StateMachine instance with the defined states and events
    sm = StateMachine[PaymentState, PaymentEvent, PaymentCtx]()

    payment_action = PaymentAction() 

    # Add transitions for the state machine

    # PROCESSING -> Authorized (Happy Path)
    sm.add_transition(
        PaymentState.PROCESSING, 
        PaymentEvent.AUTHORIZE, 
        PaymentState.AUTHORIZED, 
        payment_action.authorize,
        payment_validators_for(["account", "balance"])
    )

    # Authorized -> Captured (Happy Path)
    sm.add_transition(
        PaymentState.AUTHORIZED, 
        PaymentEvent.CAPTURE, 
        PaymentState.CAPTURED, 
        payment_action.capture,
        payment_validators_for

    )

    #  PROCESSING -> Failed (Failure Path)
    sm.add_transition(
        PaymentState.PROCESSING, 
        PaymentEvent.AUTHORIZE,
        PaymentState.FAILED,
        payment_action.fail
    )

    #  Authorize -> Failed (Failure Path)
    sm.add_transition(
        PaymentState.AUTHORIZED, 
        PaymentEvent.CAPTURE,
        PaymentState.FAILED,
        payment_action.fail
    )


    # Captured -> Refunded (Failure Path)
    sm.add_transition(
        PaymentState.CAPTURED, 
        PaymentEvent.REFUND, 
        PaymentState.REFUNDED,
        payment_action.refund
    )

    # Captured -> Cancelled (Failure Path)
    sm.add_transition(
        PaymentState.CAPTURED, 
        PaymentEvent.CANCEL, 
        PaymentState.CANCELLED, 
        payment_action.cancel
    )

    # Captured -> Completed (Happy Path)
    sm.add_transition(
        PaymentState.CAPTURED, 
        PaymentEvent.COMPLETE, 
        PaymentState.COMPLETED, 
        payment_action.complete
    )

    # Captured -> Failed (Failure Path)
    sm.add_transition(
        PaymentState.CAPTURED, 
        PaymentEvent.FAIL, 
        PaymentState.FAILED,
        payment_action.fail
    )

    # Test happy path transitions
    ctx = PaymentCtx(account_id="12345", user_id="Tommy Cruz")
    sm.set_initial_state(PaymentState.PROCESSING)
    sm.handle_event(ctx, PaymentEvent.AUTHORIZE)
    sm.handle_event(ctx, PaymentEvent.CAPTURE)
    sm.handle_event(ctx, PaymentEvent.COMPLETE)
    ctx.print_audit_log()

    # Test invalid event transition (e.g., trying to REFUND from COMPLETED state)
    ctx11 = PaymentCtx(account_id="3456", user_id="Chris Peeves")
    sm.set_initial_state(PaymentState.COMPLETED)
    try:
        sm.handle_event(ctx, PaymentEvent.REFUND)  # Invalid transition from COMPLETED
    except InvalidStateTransitionError as e:        
        print(f"Caught an error during transition: {e}")
    finally:
        ctx11.print_audit_log()   

    # Test failure path transitions
    ctx2 = PaymentCtx(txn_id="6789", usera_id="Angelica Boli")
    sm.set_initial_state(PaymentState.PROCESSING)
    sm.handle_event(ctx2, PaymentEvent.AUTHORIZE)
    sm.handle_event(ctx2, PaymentEvent.CAPTURE)
    sm.handle_event(ctx2, PaymentEvent.REFUND)
    ctx2.print_audit_log()


    # Test failure path transitions
    ctx2 = PaymentCtx(txn_id="67890")
    sm.set_initial_state(PaymentState.PROCESSING)
    sm.handle_event(ctx2, PaymentEvent.AUTHORIZE)
    sm.handle_event(ctx2, PaymentEvent.CAPTURE)
    sm.handle_event(ctx2, PaymentEvent.FAILED)
    ctx2.print_audit_log()


    # Test failure path transitions
    # PROCESSING -> Authorized -> Fail
    ctx21 = PaymentCtx(txn_id="67891")
    sm.set_initial_state(PaymentState.PROCESSING)
    sm.handle_event(ctx21, PaymentEvent.AUTHORIZE)
    sm.handle_event(ctx2, PaymentEvent.FAILED)
    ctx21.print_audit_log()

    # test cancellation path transitions
    # PROCESSING -> Authorize -> Capture -> Cancel
    ctx3 = PaymentCtx(txn_id="54321")
    sm.set_initial_state(PaymentState.PROCESSING)
    sm.handle_event(ctx3, PaymentEvent.AUTHORIZE)
    sm.handle_event(ctx3, PaymentEvent.CAPTURE)
    sm.handle_event(ctx3, PaymentEvent.CANCEL)    
    ctx3.print_audit_log()


if __name__ == "__main__":
    basic_test()
