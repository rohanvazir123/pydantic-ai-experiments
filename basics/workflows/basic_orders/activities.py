from datetime import timedelta
from temporalio import activity

@activity.defn
async def charge_payment(amount: float) -> str:
    # Perform external API/database call
    if amount <= 0:
        raise ValueError("Charge Payment amount must be greater than zero")
    return f"Charged ${amount:.2f} successfully."

@activity.defn
async def send_receipt(email_data: dict) -> str:
    # Perform email API call
    return f"Receipt sent to {email_data['email']}"

@activity.defn
async def refund_payment(amount: float) -> str:
    # Perform external API/database call
    if amount <= 0:
        raise ValueError("Refund Payment amount must be greater than zero")
    return f"Refunded ${amount:.2f} successfully."

