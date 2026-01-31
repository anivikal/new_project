"""
Main FastAPI application for Battery Smart Voicebot.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.api.routes import handoff_router, voice_router
from src.config import get_settings

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "voicebot_requests_total",
    "Total requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "voicebot_request_latency_seconds",
    "Request latency",
    ["method", "endpoint"]
)
HANDOFF_COUNT = Counter(
    "voicebot_handoffs_total",
    "Total handoffs triggered",
    ["trigger", "priority"]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    
    logger.info(
        "application_starting",
        environment=settings.environment,
        debug=settings.api.debug
    )
    
    # Initialize components
    # Pre-warm connections, caches, etc.
    
    yield
    
    # Cleanup
    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Battery Smart Voicebot API",
        description="""
        Multilingual Voicebot for Battery Smart Driver Support.
        
        ## Features
        - Real-time voice conversation (WebSocket)
        - Hindi/Hinglish natural language understanding
        - Intent detection for Tier-1 queries
        - Automatic sentiment monitoring
        - Warm handoff to human agents
        
        ## Use Cases
        - Swap history and invoice queries
        - Nearest station lookup
        - Subscription management
        - Leave and DSK activation
        """,
        version="1.0.0",
        docs_url="/docs" if settings.api.debug else None,
        redoc_url="/redoc" if settings.api.debug else None,
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging and metrics middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        start_time = time.time()
        
        # Generate request ID
        request_id = request.headers.get("X-Request-ID", str(time.time()))
        
        # Process request
        try:
            response = await call_next(request)
            
            # Record metrics
            latency = time.time() - start_time
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(latency)
            
            # Log request
            logger.info(
                "request_completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                latency_ms=int(latency * 1000)
            )
            
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            logger.error(
                "request_failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e)
            )
            raise
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # Health check endpoints
    @app.get("/health")
    async def health_check():
        """Basic health check."""
        return {"status": "healthy"}
    
    @app.get("/health/ready")
    async def readiness_check():
        """Readiness check with dependency validation."""
        checks = {
            "api": True,
            # Add more checks: database, redis, etc.
        }
        
        all_healthy = all(checks.values())
        return {
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks
        }
    
    @app.get("/health/live")
    async def liveness_check():
        """Liveness check."""
        return {"status": "alive"}
    
    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type="text/plain"
        )
    
    # Include routers
    app.include_router(voice_router, prefix="/api/v1")
    app.include_router(handoff_router, prefix="/api/v1")
    
    # Serve test UI
    @app.get("/", response_class=HTMLResponse)
    async def root():
        """Redirect to test UI."""
        return """
        <html>
            <head><title>Battery Smart Voicebot</title></head>
            <body style="font-family: Arial; text-align: center; padding: 50px; background: #1a1a2e; color: white;">
                <h1>🔋 Battery Smart Voicebot</h1>
                <p>Welcome to the Multilingual Driver Support Voicebot API</p>
                <div style="margin: 30px;">
                    <a href="/test" style="padding: 15px 30px; background: linear-gradient(135deg, #00ff88, #00d4ff); 
                       color: #1a1a2e; text-decoration: none; border-radius: 10px; font-weight: bold;">
                       Open Test UI
                    </a>
                </div>
                <p style="color: #888; font-size: 14px;">API Documentation: <a href="/docs" style="color: #00d4ff;">/docs</a></p>
            </body>
        </html>
        """
    
    @app.get("/test", response_class=HTMLResponse)
    async def test_ui():
        """Serve the test UI HTML file."""
        test_ui_path = Path(__file__).parent.parent.parent / "test_ui.html"
        if test_ui_path.exists():
            return HTMLResponse(content=test_ui_path.read_text(), status_code=200)
        return HTMLResponse(content="<h1>Test UI not found</h1>", status_code=404)
    
    return app


# Application instance
app = create_app()
