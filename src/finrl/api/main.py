"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from finrl.api.routes import router
from finrl.config import get_settings
from finrl.logging_config import configure_logging
from finrl.observability import configure_tracing, instrument_app


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(settings.service_name)

    app = FastAPI(
        title="FinRL Portfolio Decision Service",
        description=(
            "Serves a trained PPO/DQN portfolio-allocation policy over a "
            "synthetic, regime-switching market simulation. Research demo, "
            "not financial advice."
        ),
        version="0.1.0",
    )
    app.include_router(router)
    instrument_app(app)
    return app


app = create_app()
