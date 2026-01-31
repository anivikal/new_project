"""
CRM/Jarvis client for Battery Smart backend integration.
Handles driver data, swap history, subscriptions, etc.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class DriverProfile:
    """Driver profile from CRM."""
    
    driver_id: str
    name: str
    phone: str
    city: str
    registered_at: datetime
    subscription_plan: str | None
    subscription_valid_till: datetime | None
    total_swaps: int
    is_active: bool


@dataclass
class SwapRecord:
    """Battery swap record."""
    
    swap_id: str
    timestamp: datetime
    station_name: str
    battery_in: str
    battery_out: str
    amount: float
    gst: float
    total: float
    invoice_number: str


@dataclass
class Invoice:
    """Invoice details."""
    
    invoice_number: str
    date: datetime
    driver_id: str
    items: list[dict]
    subtotal: float
    gst: float
    total: float
    payment_status: str


class CRMClient:
    """
    Client for Battery Smart CRM/Jarvis API.
    
    In production, this connects to the actual backend.
    For MVP, returns mock data.
    """
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = self.settings.crm.base_url
        self.api_key = self.settings.crm.api_key
        self.timeout = self.settings.crm.timeout_seconds
    
    async def get_driver_profile(self, phone_number: str) -> DriverProfile | None:
        """Get driver profile by phone number."""
        # Mock implementation
        logger.info("crm_get_driver_profile", phone=phone_number[-4:])
        
        # In production:
        # async with httpx.AsyncClient() as client:
        #     response = await client.get(
        #         f"{self.base_url}/drivers/phone/{phone_number}",
        #         headers={"Authorization": f"Bearer {self.api_key}"},
        #         timeout=self.timeout
        #     )
        #     return DriverProfile(**response.json())
        
        return DriverProfile(
            driver_id="DRV001234",
            name="Rahul Kumar",
            phone=phone_number,
            city="Mumbai",
            registered_at=datetime(2023, 6, 15),
            subscription_plan="Monthly Premium",
            subscription_valid_till=datetime(2024, 2, 28),
            total_swaps=156,
            is_active=True
        )
    
    async def get_swap_history(
        self,
        driver_id: str,
        limit: int = 10,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> list[SwapRecord]:
        """Get driver's swap history."""
        logger.info(
            "crm_get_swap_history",
            driver_id=driver_id,
            limit=limit
        )
        
        # Mock data
        swaps = [
            SwapRecord(
                swap_id="SWP001",
                timestamp=datetime(2024, 1, 29, 10, 30),
                station_name="Andheri West Hub",
                battery_in="BAT-A12345",
                battery_out="BAT-B67890",
                amount=50.0,
                gst=9.0,
                total=59.0,
                invoice_number="INV-2024-001234"
            ),
            SwapRecord(
                swap_id="SWP002",
                timestamp=datetime(2024, 1, 28, 14, 15),
                station_name="Bandra Station",
                battery_in="BAT-B67890",
                battery_out="BAT-C11111",
                amount=50.0,
                gst=9.0,
                total=59.0,
                invoice_number="INV-2024-001233"
            ),
        ]
        
        return swaps[:limit]
    
    async def get_invoice(self, invoice_number: str) -> Invoice | None:
        """Get invoice details."""
        logger.info("crm_get_invoice", invoice=invoice_number)
        
        return Invoice(
            invoice_number=invoice_number,
            date=datetime(2024, 1, 29),
            driver_id="DRV001234",
            items=[
                {"description": "Battery Swap", "amount": 50.0},
                {"description": "GST (18%)", "amount": 9.0},
            ],
            subtotal=50.0,
            gst=9.0,
            total=59.0,
            payment_status="paid"
        )
    
    async def get_subscription_status(self, driver_id: str) -> dict:
        """Get subscription status."""
        logger.info("crm_get_subscription", driver_id=driver_id)
        
        return {
            "plan": "Monthly Premium",
            "status": "active",
            "valid_till": "2024-02-28",
            "swaps_this_month": 12,
            "unlimited_swaps": True,
            "auto_renew": True
        }
    
    async def request_leave(
        self,
        driver_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """Request subscription leave/pause."""
        logger.info(
            "crm_request_leave",
            driver_id=driver_id,
            days=(end_date - start_date).days
        )
        
        return {
            "success": True,
            "leave_id": "LV001",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "message": "Leave approved"
        }
    
    async def activate_dsk(self, driver_id: str, station_id: str) -> dict:
        """Activate DSK for new driver."""
        logger.info("crm_activate_dsk", driver_id=driver_id)
        
        return {
            "success": True,
            "activation_id": "ACT001",
            "message": "DSK activation initiated. Please visit the hub."
        }


# Singleton
_crm_client: CRMClient | None = None


def get_crm_client() -> CRMClient:
    """Get CRM client instance."""
    global _crm_client
    if _crm_client is None:
        _crm_client = CRMClient()
    return _crm_client
