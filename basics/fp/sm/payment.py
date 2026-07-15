from statemachine import StateChart, State

class PaymentWorkflow(StateChart):
    # States
    idle = State('Idle', initial=True)
    processing = State('Processing')
    completed = State('Completed')
    failed = State('Failed', final=True)

    # Standard Happy Path Transitions
    start_payment = idle.to(processing)
    success = processing.to(completed)

    # 1. Map the built-in error event to a safety/fallback transition
    error_execution = processing.to(failed)

    # Action that simulates a failure (e.g., API timeout)
    def on_start_payment(self):
        print("Attempting to connect to payment gateway...")
        raise ConnectionError("Gateway timed out!")  # This will trigger error_execution

    # 2. Extract error context inside your failure action hook
    def on_enter_failed(self, event_data=None):
        print(f"Workflow Halted!")
        if event_data and event_data.exception:
            print(f"Reason for failure: {event_data.exception}")

# Execute the workflow
sm = PaymentWorkflow()
sm.send("start_payment")

print(f"Current State: {sm.current_state.name}")
