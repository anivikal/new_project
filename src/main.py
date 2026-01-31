"""
Main entry point for Battery Smart Voicebot.
"""

import uvicorn

from src.config import get_settings


def main() -> None:
    """Run the voicebot application."""
    settings = get_settings()
    
    uvicorn.run(
        "src.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=settings.api.workers if not settings.api.debug else 1,
        reload=settings.api.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
