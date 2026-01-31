"""
Station locator service for finding nearest Battery Smart stations.
"""

from dataclasses import dataclass
from typing import Any

import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


@dataclass
class Station:
    """Battery Smart station details."""
    
    station_id: str
    name: str
    address: str
    city: str
    latitude: float
    longitude: float
    batteries_available: int
    slots_available: int
    is_open: bool
    opening_hours: str
    distance_km: float | None = None


@dataclass
class StationAvailability:
    """Real-time station availability."""
    
    station_id: str
    batteries_available: int
    slots_available: int
    last_updated: str
    wait_time_minutes: int


class StationLocatorService:
    """
    Service for finding and checking Battery Smart stations.
    
    In production, this connects to the station management system.
    """
    
    # Mock station data
    MOCK_STATIONS = [
        Station(
            station_id="ST001",
            name="Andheri West Hub",
            address="Near Metro Station, Andheri West",
            city="Mumbai",
            latitude=19.1196,
            longitude=72.8464,
            batteries_available=12,
            slots_available=8,
            is_open=True,
            opening_hours="6 AM - 10 PM"
        ),
        Station(
            station_id="ST002",
            name="Bandra Station",
            address="Linking Road, Bandra West",
            city="Mumbai",
            latitude=19.0596,
            longitude=72.8295,
            batteries_available=8,
            slots_available=5,
            is_open=True,
            opening_hours="24/7"
        ),
        Station(
            station_id="ST003",
            name="Powai Hub",
            address="Hiranandani Gardens, Powai",
            city="Mumbai",
            latitude=19.1175,
            longitude=72.9060,
            batteries_available=15,
            slots_available=10,
            is_open=True,
            opening_hours="6 AM - 11 PM"
        ),
    ]
    
    def __init__(self) -> None:
        self.settings = get_settings()
    
    async def find_nearest_stations(
        self,
        latitude: float | None = None,
        longitude: float | None = None,
        city: str | None = None,
        limit: int = 5
    ) -> list[Station]:
        """Find nearest stations based on location or city."""
        logger.info(
            "finding_nearest_stations",
            lat=latitude,
            lng=longitude,
            city=city
        )
        
        stations = self.MOCK_STATIONS.copy()
        
        # Filter by city if provided
        if city:
            stations = [s for s in stations if s.city.lower() == city.lower()]
        
        # Calculate distances if coordinates provided
        if latitude and longitude:
            for station in stations:
                station.distance_km = self._calculate_distance(
                    latitude, longitude,
                    station.latitude, station.longitude
                )
            
            # Sort by distance
            stations.sort(key=lambda s: s.distance_km or float('inf'))
        
        return stations[:limit]
    
    async def get_station_availability(self, station_id: str) -> StationAvailability | None:
        """Get real-time availability for a station."""
        logger.info("checking_station_availability", station_id=station_id)
        
        station = next((s for s in self.MOCK_STATIONS if s.station_id == station_id), None)
        if not station:
            return None
        
        return StationAvailability(
            station_id=station_id,
            batteries_available=station.batteries_available,
            slots_available=station.slots_available,
            last_updated="2 minutes ago",
            wait_time_minutes=2
        )
    
    async def search_stations_by_name(self, query: str) -> list[Station]:
        """Search stations by name."""
        query_lower = query.lower()
        
        return [
            s for s in self.MOCK_STATIONS
            if query_lower in s.name.lower() or query_lower in s.address.lower()
        ]
    
    def _calculate_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in kilometers."""
        import math
        
        # Haversine formula
        R = 6371  # Earth's radius in km
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return round(R * c, 1)


# Singleton
_station_service: StationLocatorService | None = None


def get_station_service() -> StationLocatorService:
    """Get station service instance."""
    global _station_service
    if _station_service is None:
        _station_service = StationLocatorService()
    return _station_service
