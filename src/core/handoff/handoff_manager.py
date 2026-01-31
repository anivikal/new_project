"""
Handoff Manager - Coordinates warm handoff process.
Manages the transition from bot to human agent.
"""

import asyncio
from datetime import datetime
from typing import Callable
from uuid import UUID

import structlog

from src.config import get_settings
from src.core.handoff.summary_generator import SummaryGenerator
from src.models import (
    ConversationSession,
    ConversationState,
    HandoffAlert,
    HandoffPriority,
    HandoffStatus,
    HandoffTrigger,
)

logger = structlog.get_logger(__name__)


class HandoffQueue:
    """
    Queue management for handoff requests.
    In production, this would use Redis or a message queue.
    """
    
    def __init__(self) -> None:
        self._queue: list[HandoffAlert] = []
        self._handlers: list[Callable[[HandoffAlert], None]] = []
    
    def add(self, alert: HandoffAlert) -> int:
        """Add handoff alert to queue."""
        # Assign queue position
        alert.queue_position = len(self._queue) + 1
        
        # Estimate wait time (mock: 30 seconds per item)
        alert.estimated_wait_seconds = alert.queue_position * 30
        
        # Sort by priority
        self._queue.append(alert)
        self._queue.sort(key=lambda x: (
            -{"urgent": 4, "high": 3, "medium": 2, "low": 1}[x.priority.value],
            x.created_at
        ))
        
        # Update positions
        for i, item in enumerate(self._queue):
            item.queue_position = i + 1
        
        # Notify handlers
        for handler in self._handlers:
            handler(alert)
        
        return alert.queue_position
    
    def get_next(self) -> HandoffAlert | None:
        """Get next handoff alert from queue."""
        if self._queue:
            alert = self._queue.pop(0)
            alert.status = HandoffStatus.ASSIGNED
            alert.assigned_at = datetime.utcnow()
            return alert
        return None
    
    def get_by_id(self, alert_id: UUID) -> HandoffAlert | None:
        """Get specific handoff alert."""
        return next((a for a in self._queue if a.id == alert_id), None)
    
    def remove(self, alert_id: UUID) -> bool:
        """Remove alert from queue."""
        for i, alert in enumerate(self._queue):
            if alert.id == alert_id:
                self._queue.pop(i)
                return True
        return False
    
    def on_new_alert(self, handler: Callable[[HandoffAlert], None]) -> None:
        """Register handler for new alerts."""
        self._handlers.append(handler)
    
    @property
    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)
    
    def get_stats(self) -> dict:
        """Get queue statistics."""
        if not self._queue:
            return {
                "total": 0,
                "by_priority": {},
                "avg_wait_seconds": 0
            }
        
        by_priority = {}
        for alert in self._queue:
            by_priority[alert.priority.value] = by_priority.get(alert.priority.value, 0) + 1
        
        return {
            "total": len(self._queue),
            "by_priority": by_priority,
            "avg_wait_seconds": sum(a.estimated_wait_seconds or 0 for a in self._queue) / len(self._queue)
        }


class HandoffNotifier:
    """
    Notification service for handoff alerts.
    Sends alerts to agent dashboard, webhooks, etc.
    """
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self._webhooks: list[str] = []
        self._websocket_handlers: list[Callable] = []
    
    async def notify_new_handoff(self, alert: HandoffAlert) -> None:
        """Send notification for new handoff."""
        logger.info(
            "handoff_notification",
            alert_id=str(alert.id),
            priority=alert.priority.value,
            trigger=alert.trigger.value
        )
        
        # Send to webhooks
        await self._send_webhooks(alert)
        
        # Notify websocket listeners (agent dashboard)
        await self._notify_websockets(alert)
    
    async def _send_webhooks(self, alert: HandoffAlert) -> None:
        """Send to configured webhooks."""
        import httpx
        
        if not self._webhooks:
            return
        
        payload = {
            "event": "new_handoff",
            "alert_id": str(alert.id),
            "conversation_id": str(alert.conversation_id),
            "call_id": alert.call_id,
            "priority": alert.priority.value,
            "trigger": alert.trigger.value,
            "driver_phone": alert.driver_info.phone_number[-4:],  # Last 4 digits only
            "summary": alert.issue_summary,
            "timestamp": alert.created_at.isoformat()
        }
        
        async with httpx.AsyncClient() as client:
            for webhook in self._webhooks:
                try:
                    await client.post(
                        webhook,
                        json=payload,
                        timeout=5.0
                    )
                except Exception as e:
                    logger.warning("webhook_failed", url=webhook, error=str(e))
    
    async def _notify_websockets(self, alert: HandoffAlert) -> None:
        """Notify connected websocket clients."""
        for handler in self._websocket_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.warning("websocket_notification_failed", error=str(e))
    
    def register_websocket_handler(self, handler: Callable) -> None:
        """Register websocket notification handler."""
        self._websocket_handlers.append(handler)
    
    def add_webhook(self, url: str) -> None:
        """Add webhook URL."""
        self._webhooks.append(url)


class HandoffManager:
    """
    Main manager for warm handoff process.
    
    Coordinates:
    1. Summary generation
    2. Alert creation
    3. Queue management
    4. Agent notification
    5. Call transfer
    """
    
    def __init__(self) -> None:
        self.settings = get_settings()
        self.summary_generator = SummaryGenerator()
        self.queue = HandoffQueue()
        self.notifier = HandoffNotifier()
        
        # Active handoffs
        self._active_handoffs: dict[str, HandoffAlert] = {}  # call_id -> alert
    
    async def initiate_handoff(
        self,
        session: ConversationSession,
        trigger: HandoffTrigger
    ) -> HandoffAlert:
        """
        Initiate warm handoff process.
        
        Steps:
        1. Generate conversation summary
        2. Create handoff alert with micro-brief
        3. Add to queue
        4. Notify agents
        5. Return alert for call handling
        
        Args:
            session: Current conversation session
            trigger: What triggered the handoff
        
        Returns:
            HandoffAlert with all details
        """
        logger.info(
            "initiating_handoff",
            session_id=str(session.id),
            call_id=session.call_id,
            trigger=trigger.value
        )
        
        # Update session state
        session.state = ConversationState.HANDOFF_PENDING
        session.handoff_triggered = True
        session.handoff_trigger = trigger
        
        # Generate summary
        summary = await self.summary_generator.generate_summary(session, trigger)
        
        # Create handoff alert
        alert = HandoffAlert.from_conversation(session, trigger, summary)
        
        # Add to queue
        queue_position = self.queue.add(alert)
        
        logger.info(
            "handoff_queued",
            alert_id=str(alert.id),
            queue_position=queue_position,
            priority=alert.priority.value
        )
        
        # Notify agents
        await self.notifier.notify_new_handoff(alert)
        
        # Store active handoff
        self._active_handoffs[session.call_id] = alert
        
        return alert
    
    async def assign_agent(
        self,
        alert_id: UUID,
        agent_id: str
    ) -> HandoffAlert | None:
        """Assign an agent to handle the handoff."""
        alert = self.queue.get_by_id(alert_id)
        if not alert:
            # Check active handoffs
            alert = next(
                (a for a in self._active_handoffs.values() if a.id == alert_id),
                None
            )
        
        if alert:
            alert.assigned_agent_id = agent_id
            alert.assigned_at = datetime.utcnow()
            alert.status = HandoffStatus.ASSIGNED
            
            # Remove from queue
            self.queue.remove(alert_id)
            
            logger.info(
                "handoff_assigned",
                alert_id=str(alert_id),
                agent_id=agent_id
            )
            
            return alert
        
        return None
    
    async def start_handoff_call(
        self,
        alert_id: UUID
    ) -> dict:
        """
        Start the actual call transfer.
        
        Returns transfer instructions for telephony system.
        """
        alert = next(
            (a for a in self._active_handoffs.values() if a.id == alert_id),
            None
        )
        
        if not alert:
            raise ValueError(f"Handoff alert not found: {alert_id}")
        
        if not alert.assigned_agent_id:
            raise ValueError("No agent assigned to handoff")
        
        alert.status = HandoffStatus.IN_PROGRESS
        
        return {
            "action": "transfer_call",
            "call_id": alert.call_id,
            "target_agent_id": alert.assigned_agent_id,
            "context": {
                "summary": alert.issue_summary,
                "priority": alert.priority.value,
                "driver_language": alert.driver_info.preferred_language.value,
            }
        }
    
    async def complete_handoff(
        self,
        alert_id: UUID,
        resolution: str | None = None
    ) -> None:
        """Mark handoff as completed."""
        # Find alert
        call_id = None
        for cid, alert in self._active_handoffs.items():
            if alert.id == alert_id:
                call_id = cid
                break
        
        if call_id:
            alert = self._active_handoffs[call_id]
            alert.status = HandoffStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            
            logger.info(
                "handoff_completed",
                alert_id=str(alert_id),
                resolution=resolution,
                duration_seconds=(alert.resolved_at - alert.created_at).total_seconds()
            )
            
            # Cleanup
            del self._active_handoffs[call_id]
    
    async def cancel_handoff(
        self,
        call_id: str,
        reason: str = "cancelled"
    ) -> None:
        """Cancel a pending handoff."""
        if call_id in self._active_handoffs:
            alert = self._active_handoffs[call_id]
            alert.status = HandoffStatus.ABANDONED
            
            # Remove from queue
            self.queue.remove(alert.id)
            
            logger.info(
                "handoff_cancelled",
                call_id=call_id,
                reason=reason
            )
            
            del self._active_handoffs[call_id]
    
    def get_handoff_status(self, call_id: str) -> dict | None:
        """Get current handoff status for a call."""
        alert = self._active_handoffs.get(call_id)
        if not alert:
            return None
        
        return {
            "alert_id": str(alert.id),
            "status": alert.status.value,
            "priority": alert.priority.value,
            "queue_position": alert.queue_position,
            "estimated_wait_seconds": alert.estimated_wait_seconds,
            "assigned_agent": alert.assigned_agent_id,
        }
    
    def get_agent_brief(self, alert_id: UUID) -> dict | None:
        """Get the micro-brief for agent display."""
        for alert in self._active_handoffs.values():
            if alert.id == alert_id:
                brief = alert.micro_brief
                return {
                    "driver_name": brief.driver_name,
                    "driver_phone_last_4": brief.driver_phone_last_4,
                    "driver_city": brief.driver_city,
                    "language": brief.driver_language.value,
                    "top_entities": brief.top_entities,
                    "summary": brief.actionable_summary,
                    "escalation_reason": brief.escalation_reason.value,
                    "escalation_description": brief.escalation_description,
                    "sentiment": brief.current_sentiment.value,
                    "sentiment_score": brief.sentiment_score,
                    "suggested_actions": [
                        {"action": a.action, "description": a.description}
                        for a in brief.suggested_actions
                    ],
                    "confidence_trend": brief.confidence_timeline.trend,
                }
        
        return None
    
    def get_queue_stats(self) -> dict:
        """Get queue statistics."""
        return self.queue.get_stats()


# Singleton instance
_handoff_manager: HandoffManager | None = None


def get_handoff_manager() -> HandoffManager:
    """Get or create handoff manager instance."""
    global _handoff_manager
    
    if _handoff_manager is None:
        _handoff_manager = HandoffManager()
    
    return _handoff_manager
