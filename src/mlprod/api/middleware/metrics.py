"""Freely adapted from: https://github.com/kozhushman/prometheusrock"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response
from starlette import status

from prometheus_client import (
    REGISTRY,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    generate_latest,
)
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_client import Counter, Histogram

from time import time

import logging
import os

LOGGER = logging.getLogger("mlprod.api.middleware.metrics")


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware that track and store the metrics for Prometheus."""

    def __init__(
        self,
        app: ASGIApp,
        app_name: str = "ASGIApp",
        skip_paths: list[str] = ["/metrics", "/docs", "/favicon.ico"],
    ) -> None:
        """Constructor."""
        super().__init__(app)
        labels = ["method", "path", "status_code", "app_name"]

        # counts all requests for api
        self.request_counter = Counter("api_request", "Total HTTP requests", labels)
        # counts exceptions for api
        self.request_exception = Counter(
            "api_exception",
            "Total number of exceptions on API requests",
            labels,
        )
        # track execution time for each request
        self.request_time = Histogram(
            "api_processing_time", "HTTP request processing time in seconds", labels
        )
        # track number of active inferences
        self.active_inferences = Gauge(
            "api_active_inferences",
            "Number of active inferences",
        )

        self.app_name = app_name
        self.skip_paths = skip_paths

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Dispatch function that process each request."""
        path = request.url.path

        if path in self.skip_paths:
            try:
                return await call_next(request)
            except Exception as e:
                LOGGER.exception("Exception on skipped path: %s", path)
                raise e

        # all paths have 3 slash then an id, here we clean the path from the id
        tokens = path.split("/")
        if len(tokens) > 3:
            tokens = tokens[:3]
        path = "/".join(tokens)

        method = request.method
        status_code = status.HTTP_408_REQUEST_TIMEOUT

        begin = time()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # track exceptions
            LOGGER.exception(
                f"Exception on path={path} method={method} app={self.app_name}: {str(e)}"
            )
            self.request_exception.labels(
                method=method, path=path, app_name=self.app_name, exception=str(e)
            )
            raise e

        finally:
            # track active inferences
            if path == "/inference/start":
                self.active_inferences.inc(1)
            if path == "/inference/results":
                self.active_inferences.dec(1)

            # track spent time
            spent_time = time() - begin

            # track visits
            labels = {
                "method": method,
                "path": path,
                "status_code": status_code,
                "app_name": self.app_name,
            }

            self.request_counter.labels(**labels).inc()
            self.request_time.labels(**labels).observe(spent_time)

        return response


def metrics_route(ignored: Request) -> Response:
    """Generate route for the /metric endpoint."""
    registry = REGISTRY
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        registry = CollectorRegistry()
        MultiProcessCollector(registry)

    data = generate_latest(registry)
    response_headers = {
        "Content-Type": CONTENT_TYPE_LATEST,
        "Content-Length": str(len(data)),
    }

    return Response(data, status_code=status.HTTP_200_OK, headers=response_headers)
