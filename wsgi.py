"""WSGI entry point for WILDLIFE-AI Dashboard."""

from __future__ import annotations

import os
from dashboard.app import create_app

# Set default password for remote binds if not already configured
if "WILDLIFE_DASHBOARD_PASSWORD" not in os.environ:
    os.environ["WILDLIFE_DASHBOARD_PASSWORD"] = "admin"

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
