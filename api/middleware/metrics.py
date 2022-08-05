"""
Freely adapted from: https://github.com/kozhushman/prometheusrock
"""
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import Response
from starlette import status

from prometheus_client import REGISTRY, CONTENT_TYPE_LATEST, CollectorRegistry, Gauge, generate_latest
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_client import Counter, Histogram

from time import time

import os


class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, 
            app: ASGIApp, 
            app_name:str='ASGIApp',
            skip_paths: list[str]=['/metrics', '/docs', '/favicon.ico']
    ) -> None:
        super().__init__(app)
        labels = ['method', 'path', 'status_code', 'app_name']

        self.request_counter = Counter(
            'api_request', 
            'Total HTTP requests',
            labels
        )
        self.request_exception = Counter(
            'api_exception',
            'Total number of exceptions on API requests',
            labels,
        )
        self.request_time = Histogram(
            'api_processing_time',
            'HTTP request processing time in seconds', 
            labels
        )
        self.active_inferences = Gauge(
            'api_active_inferences',
            'Number of active inferences',
        )

        self.app_name = app_name
        self.skip_paths = skip_paths
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        if path in self.skip_paths:
            try:
                return await call_next(request)
            except Exception as e:
                    raise e

        # all paths have 3 slash then an id, here we clean the path from the id
        tokens = path.split('/')
        if len(tokens) > 3:
            tokens = tokens[:3]
        path = '/'.join(tokens)

        method = request.method
        status_code = status.HTTP_408_REQUEST_TIMEOUT

        begin = time()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # track exceptions
            self.request_exception.labels(method=method, path=path, app_name=self.app_name, exception=str(e))
            raise e

        finally:
            # track active inferences
            if path == '/inference/start':
                self.active_inferences.inc(1)
            if path == '/inference/results':
                self.active_inferences.dec(1)

            # track spent time
            spent_time = time() - begin

            # track visits
            labels = {
                'method': method,
                'path': path,
                'status_code': status_code,
                'app_name': self.app_name,
            }

            self.request_counter.labels(**labels).inc()
            self.request_time.labels(**labels).observe(spent_time)
        
        return response


def metrics_route(ignored: Request) -> Response:
    registry = REGISTRY
    if 'PROMETHEUS_MULTIPROC_DIR' in os.environ:
        registry = CollectorRegistry()
        MultiProcessCollector(registry)
    
    data = generate_latest(registry)
    response_headers = {
        'Content-Type': CONTENT_TYPE_LATEST,
        'Content-Length': str(len(data))
    }

    return Response(data, status_code=status.HTTP_200_OK, headers=response_headers)
