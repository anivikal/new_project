"""
Handoff API routes for agent dashboard integration.
Provides endpoints for managing handoff queue and agent assignments.
"""

from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from src.core.handoff import get_handoff_manager
from src.models import HandoffPriority, HandoffStatus

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/handoff", tags=["Handoff"])


# Request/Response Models
class AssignAgentRequest(BaseModel):
    """Request to assign agent to handoff."""
    
    alert_id: str
    agent_id: str


class CompleteHandoffRequest(BaseModel):
    """Request to complete a handoff."""
    
    alert_id: str
    resolution: str | None = None
    notes: str | None = None


class HandoffAlertResponse(BaseModel):
    """Handoff alert summary for API response."""
    
    id: str
    conversation_id: str
    call_id: str
    trigger: str
    priority: str
    status: str
    driver_phone_last_4: str
    driver_city: str | None
    driver_language: str
    issue_summary: str
    queue_position: int | None
    estimated_wait_seconds: int | None
    assigned_agent_id: str | None
    created_at: str


class AgentBriefResponse(BaseModel):
    """Agent micro-brief for quick context."""
    
    driver_name: str | None
    driver_phone_last_4: str
    driver_city: str | None
    language: str
    top_entities: dict
    summary: str
    escalation_reason: str
    escalation_description: str
    sentiment: str
    sentiment_score: float
    suggested_actions: list[dict]
    confidence_trend: str


class QueueStatsResponse(BaseModel):
    """Queue statistics."""
    
    total: int
    by_priority: dict[str, int]
    avg_wait_seconds: float


@router.get("/queue", response_model=list[HandoffAlertResponse])
async def get_handoff_queue():
    """
    Get all pending handoff alerts in queue.
    
    Returns alerts sorted by priority (urgent first) and creation time.
    """
    handoff_manager = get_handoff_manager()
    
    alerts = []
    for alert in handoff_manager.queue._queue:
        alerts.append(HandoffAlertResponse(
            id=str(alert.id),
            conversation_id=str(alert.conversation_id),
            call_id=alert.call_id,
            trigger=alert.trigger.value,
            priority=alert.priority.value,
            status=alert.status.value,
            driver_phone_last_4=alert.driver_info.phone_number[-4:],
            driver_city=alert.driver_info.city,
            driver_language=alert.driver_info.preferred_language.value,
            issue_summary=alert.issue_summary,
            queue_position=alert.queue_position,
            estimated_wait_seconds=alert.estimated_wait_seconds,
            assigned_agent_id=alert.assigned_agent_id,
            created_at=alert.created_at.isoformat()
        ))
    
    return alerts


@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """Get queue statistics."""
    handoff_manager = get_handoff_manager()
    stats = handoff_manager.get_queue_stats()
    
    return QueueStatsResponse(
        total=stats["total"],
        by_priority=stats["by_priority"],
        avg_wait_seconds=stats["avg_wait_seconds"]
    )


@router.get("/alert/{alert_id}")
async def get_handoff_alert(alert_id: str):
    """Get detailed handoff alert information."""
    handoff_manager = get_handoff_manager()
    
    alert_uuid = UUID(alert_id)
    
    # Search in queue and active handoffs
    alert = handoff_manager.queue.get_by_id(alert_uuid)
    if not alert:
        for active in handoff_manager._active_handoffs.values():
            if active.id == alert_uuid:
                alert = active
                break
    
    if not alert:
        raise HTTPException(status_code=404, detail="Handoff alert not found")
    
    return {
        "id": str(alert.id),
        "conversation_id": str(alert.conversation_id),
        "call_id": alert.call_id,
        "trigger": alert.trigger.value,
        "trigger_description": alert.trigger_description,
        "priority": alert.priority.value,
        "status": alert.status.value,
        "driver_info": {
            "phone_last_4": alert.driver_info.phone_number[-4:],
            "name": alert.driver_info.name,
            "city": alert.driver_info.city,
            "language": alert.driver_info.preferred_language.value,
            "subscription_plan": alert.driver_info.subscription_plan,
        },
        "intent_history": [i.value for i in alert.intent_history],
        "current_intent": alert.current_intent.value if alert.current_intent else None,
        "sentiment": alert.sentiment.value,
        "sentiment_score": alert.sentiment_score,
        "issue_summary": alert.issue_summary,
        "detailed_summary": {
            "one_line": alert.detailed_summary.one_line_summary,
            "detailed": alert.detailed_summary.detailed_summary,
            "primary_issue": alert.detailed_summary.primary_issue,
            "secondary_issues": alert.detailed_summary.secondary_issues,
            "stuck_on": alert.detailed_summary.stuck_on,
            "topics_discussed": alert.detailed_summary.topics_discussed,
        },
        "actions_taken": [
            {"action": a.action, "description": a.description, "success": a.success}
            for a in alert.actions_taken_by_bot
        ],
        "suggested_actions": [
            {"action": a.action, "description": a.description, "priority": a.priority}
            for a in alert.next_steps_for_agent
        ],
        "conversation_turns": [
            {
                "role": t.role.value,
                "content": t.content,
                "timestamp": t.timestamp.isoformat(),
                "intent": t.nlu_result.intent.intent.value if t.nlu_result else None,
                "sentiment": t.nlu_result.sentiment.label.value if t.nlu_result else None,
            }
            for t in alert.conversation_turns
        ],
        "queue_position": alert.queue_position,
        "assigned_agent_id": alert.assigned_agent_id,
        "created_at": alert.created_at.isoformat(),
        "assigned_at": alert.assigned_at.isoformat() if alert.assigned_at else None,
    }


@router.get("/alert/{alert_id}/brief", response_model=AgentBriefResponse)
async def get_agent_brief(alert_id: str):
    """
    Get the micro-brief for agent display.
    
    This is the quick-glance view shown to agents when they accept a handoff.
    """
    handoff_manager = get_handoff_manager()
    
    brief = handoff_manager.get_agent_brief(UUID(alert_id))
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")
    
    return AgentBriefResponse(**brief)


@router.post("/assign")
async def assign_agent(request: AssignAgentRequest):
    """
    Assign an agent to handle a handoff.
    
    This removes the alert from the queue and marks it as assigned.
    """
    handoff_manager = get_handoff_manager()
    
    alert = await handoff_manager.assign_agent(
        alert_id=UUID(request.alert_id),
        agent_id=request.agent_id
    )
    
    if not alert:
        raise HTTPException(status_code=404, detail="Handoff alert not found")
    
    logger.info(
        "agent_assigned",
        alert_id=request.alert_id,
        agent_id=request.agent_id
    )
    
    return {
        "status": "assigned",
        "alert_id": str(alert.id),
        "agent_id": alert.assigned_agent_id,
        "call_id": alert.call_id
    }


@router.post("/start/{alert_id}")
async def start_handoff_call(alert_id: str):
    """
    Start the actual call transfer to agent.
    
    Returns transfer instructions for the telephony system.
    """
    handoff_manager = get_handoff_manager()
    
    try:
        transfer_info = await handoff_manager.start_handoff_call(UUID(alert_id))
        
        logger.info("handoff_call_started", alert_id=alert_id)
        
        return transfer_info
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/complete")
async def complete_handoff(request: CompleteHandoffRequest):
    """
    Mark a handoff as completed.
    
    Called when the agent has resolved the driver's issue.
    """
    handoff_manager = get_handoff_manager()
    
    await handoff_manager.complete_handoff(
        alert_id=UUID(request.alert_id),
        resolution=request.resolution
    )
    
    logger.info(
        "handoff_completed",
        alert_id=request.alert_id,
        resolution=request.resolution
    )
    
    return {"status": "completed", "alert_id": request.alert_id}


@router.get("/status/{call_id}")
async def get_handoff_status(call_id: str):
    """
    Get handoff status for a specific call.
    
    Used by telephony system to check if call is in handoff.
    """
    handoff_manager = get_handoff_manager()
    
    status = handoff_manager.get_handoff_status(call_id)
    if not status:
        return {"in_handoff": False, "call_id": call_id}
    
    return {
        "in_handoff": True,
        "call_id": call_id,
        **status
    }


# WebSocket for real-time agent dashboard updates
class AgentDashboardManager:
    """Manages WebSocket connections for agent dashboards."""
    
    def __init__(self) -> None:
        self.connections: dict[str, WebSocket] = {}  # agent_id -> websocket
    
    async def connect(self, agent_id: str, websocket: WebSocket) -> None:
        """Register agent dashboard connection."""
        await websocket.accept()
        self.connections[agent_id] = websocket
        logger.info("agent_dashboard_connected", agent_id=agent_id)
    
    def disconnect(self, agent_id: str) -> None:
        """Remove agent dashboard connection."""
        if agent_id in self.connections:
            del self.connections[agent_id]
            logger.info("agent_dashboard_disconnected", agent_id=agent_id)
    
    async def broadcast_new_alert(self, alert_data: dict) -> None:
        """Broadcast new handoff alert to all connected agents."""
        message = {
            "type": "new_alert",
            "data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        disconnected = []
        for agent_id, ws in self.connections.items():
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(agent_id)
        
        for agent_id in disconnected:
            self.disconnect(agent_id)
    
    async def notify_agent(self, agent_id: str, message: dict) -> None:
        """Send message to specific agent."""
        if agent_id in self.connections:
            try:
                await self.connections[agent_id].send_json(message)
            except Exception:
                self.disconnect(agent_id)


dashboard_manager = AgentDashboardManager()


@router.websocket("/dashboard/{agent_id}")
async def agent_dashboard_ws(websocket: WebSocket, agent_id: str):
    """
    WebSocket for real-time agent dashboard updates.
    
    Receives:
    - New handoff alerts
    - Queue updates
    - Assignment notifications
    """
    await dashboard_manager.connect(agent_id, websocket)
    
    # Register with handoff manager for notifications
    handoff_manager = get_handoff_manager()
    
    async def on_new_alert(alert):
        await dashboard_manager.broadcast_new_alert({
            "id": str(alert.id),
            "priority": alert.priority.value,
            "trigger": alert.trigger.value,
            "summary": alert.issue_summary,
            "driver_phone_last_4": alert.driver_info.phone_number[-4:],
        })
    
    handoff_manager.notifier.register_websocket_handler(on_new_alert)
    
    try:
        # Send current queue status
        queue_alerts = []
        for alert in handoff_manager.queue._queue:
            queue_alerts.append({
                "id": str(alert.id),
                "priority": alert.priority.value,
                "trigger": alert.trigger.value,
                "summary": alert.issue_summary,
                "queue_position": alert.queue_position,
            })
        
        await websocket.send_json({
            "type": "queue_sync",
            "data": queue_alerts,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_json()
                
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif data.get("type") == "accept":
                    # Agent accepting a handoff
                    alert_id = data.get("alert_id")
                    if alert_id:
                        alert = await handoff_manager.assign_agent(
                            UUID(alert_id),
                            agent_id
                        )
                        if alert:
                            brief = handoff_manager.get_agent_brief(UUID(alert_id))
                            await websocket.send_json({
                                "type": "assignment_confirmed",
                                "data": {
                                    "alert_id": alert_id,
                                    "brief": brief
                                },
                                "timestamp": datetime.utcnow().isoformat()
                            })
            
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("dashboard_ws_error", error=str(e))
                break
    
    finally:
        dashboard_manager.disconnect(agent_id)
