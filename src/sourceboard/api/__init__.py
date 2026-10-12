"""API server: FastAPI routes, service lifecycle, and static web UI."""

from .api import LOCAL_HOSTS, WEB_DIR, Modality, Role, Services, create_app

__all__ = ["LOCAL_HOSTS", "Modality", "Role", "Services", "WEB_DIR", "create_app"]
