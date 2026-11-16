"""Flask routes for the local WILDLIFE-AI management dashboard."""

from __future__ import annotations

import hmac
import logging
import os
import time
from pathlib import Path
from typing import Any, Protocol

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
    url_for,
)

import config
from dashboard.controller import DashboardController

logger = logging.getLogger("dashboard.web")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_AUDIO_PASSWORD_ENV = "WILDLIFE_DASHBOARD_PASSWORD"


class DashboardDatabase(Protocol):
    def list_recent_events(self, limit: int = 100) -> list[dict[str, object]]: ...


class DashboardService(Protocol):
    @property
    def database(self) -> DashboardDatabase: ...

    @property
    def latest_frame(self) -> bytes | None: ...

    def status(self) -> dict[str, object]: ...

    def start(self) -> bool: ...

    def stop(self) -> bool: ...

    def emergency_stop(self) -> None: ...

    def set_deterrence_enabled(self, enabled: bool) -> None: ...

    def test_sound(self) -> None: ...


def create_app(
    controller: DashboardService | None = None,
    *,
    host: str | None = None,
) -> Flask:
    """Create the dashboard app; remote binds require a configured password."""
    bind_host = config.DASHBOARD_HOST if host is None else host
    password = os.environ.get(_AUDIO_PASSWORD_ENV, "")
    if bind_host.casefold() not in _LOOPBACK_HOSTS and not password:
        raise RuntimeError(
            f"Set {_AUDIO_PASSWORD_ENV} before binding the dashboard beyond localhost."
        )

    dashboard = Flask(__name__)
    service = DashboardController() if controller is None else controller
    dashboard.extensions["wildlife_controller"] = service
    dashboard.extensions["wildlife_password"] = password

    @dashboard.before_request
    def require_dashboard_password() -> Response | None:
        expected = dashboard.extensions["wildlife_password"]
        if not expected:
            return None
        authorization = request.authorization
        supplied = "" if authorization is None else authorization.password or ""
        if hmac.compare_digest(supplied, expected):
            return None
        return Response(
            "Authentication required.",
            status=401,
            headers={"WWW-Authenticate": 'Basic realm="WILDLIFE-AI"'},
        )

    @dashboard.get("/")
    def index() -> str:
        return render_template("index.html")

    @dashboard.get("/video_feed")
    def video_feed() -> Response:
        def generate():
            while True:
                frame = service.latest_frame
                if frame is None:
                    time.sleep(0.25)
                    continue
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
                time.sleep(1.0 / config.DASHBOARD_STREAM_FPS)

        return Response(
            stream_with_context(generate()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @dashboard.get("/api/status")
    def status() -> Response:
        return jsonify(
            status=service.status(),
            source=config.CAMERA_SOURCE,
            detector=config.DASHBOARD_DETECTOR,
            cooldown_seconds=config.DETERRENCE_COOLDOWN_SECONDS,
        )

    @dashboard.get("/api/events")
    def events() -> Response | tuple[Response, int]:
        try:
            limit = int(request.args.get("limit", "50"))
        except ValueError:
            return jsonify(error="limit must be an integer."), 400
        if not 1 <= limit <= 200:
            return jsonify(error="limit must be between 1 and 200."), 400

        rows = service.database.list_recent_events(limit)
        snapshot_dir = Path(config.SNAPSHOTS_DIR).resolve()
        for row in rows:
            stored_path = row.pop("snapshot_path", None)
            row["snapshot_url"] = None
            if stored_path:
                image_path = Path(str(stored_path)).resolve()
                if image_path.parent == snapshot_dir and image_path.is_file():
                    row["snapshot_url"] = url_for(
                        "snapshot_image",
                        filename=image_path.name,
                    )
        return jsonify(events=rows)

    @dashboard.get("/snapshots/<path:filename>")
    def snapshot_image(filename: str) -> Response:
        if Path(filename).name != filename:
            abort(404)
        snapshot_dir = Path(config.SNAPSHOTS_DIR).resolve()
        image_path = (snapshot_dir / filename).resolve()
        if image_path.parent != snapshot_dir or not image_path.is_file():
            abort(404)
        return send_file(image_path, mimetype="image/jpeg", conditional=True)

    @dashboard.post("/api/system/start")
    def start_system() -> Response | tuple[Response, int]:
        try:
            started = service.start()
        except RuntimeError as exc:
            logger.exception("Could not start dashboard monitoring.")
            return jsonify(error=str(exc)), 409
        return jsonify(started=started, status=service.status()), 202 if started else 200

    @dashboard.post("/api/system/stop")
    def stop_system() -> Response | tuple[Response, int]:
        stopped = service.stop()
        return jsonify(stopped=stopped, status=service.status()), 200 if stopped else 202

    @dashboard.post("/api/emergency-stop")
    def emergency_stop() -> Response:
        service.emergency_stop()
        return jsonify(status=service.status())

    @dashboard.post("/api/deterrence")
    def set_deterrence() -> Response | tuple[Response, int]:
        data: Any = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("enabled"), bool):
            return jsonify(error='JSON body must contain a boolean "enabled".'), 400
        try:
            service.set_deterrence_enabled(data["enabled"])
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 409
        return jsonify(status=service.status())

    @dashboard.post("/api/test-sound")
    def test_sound() -> Response | tuple[Response, int]:
        try:
            service.test_sound()
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("Dashboard sound test was rejected: %s", exc)
            return jsonify(error=str(exc)), 400
        except (OSError, RuntimeError) as exc:
            logger.exception("Dashboard sound test failed.")
            return jsonify(error=str(exc)), 409
        return jsonify(message="Sound test started.")

    return dashboard


def main() -> None:
    from main import configure_logging

    configure_logging()
    app = create_app()
    service: DashboardController = app.extensions["wildlife_controller"]
    try:
        app.run(
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            threaded=True,
            use_reloader=False,
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
