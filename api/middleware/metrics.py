from datadog import DogStatsd
from timing_asgi import TimingClient

import os


STATSD_URL = os.environ.get('STATSD_URL')
STATSD_PORT = int(os.environ.get('STATSD_PORT'))

statsd = DogStatsd(host=STATSD_URL, port=STATSD_PORT)


class Timings(TimingClient):
    def __init__(self) -> None:
        super().__init__()

    def timing(self, metric_name, timing, tags):
        print(metric_name, timing, tags)
