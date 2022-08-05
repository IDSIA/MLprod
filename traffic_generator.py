from time import sleep

from api.requests import LocationData, UserData
from datas import read_user_config, generate_user_data_from_config, UserLabeller

import multiprocessing
import signal
import numpy as np
import requests
import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)5s %(message)s',
)


def setup_arguments():
    """Defines the available input parameters"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', help='Set fixed seed', default=42, type=int)
    parser.add_argument('-n', help='Number of requests to do, 0 means infinite', default=None, type=int)
    parser.add_argument('--no-sleep', help='If set, no sleeps are used', default=False, action='store_true')
    parser.add_argument('--config', help='User config file to use', default='./config/user.tsv', type=str)
    parser.add_argument('-p', help='Number of parallel generators.', default=1, type=int)
    parser.add_argument('-a', help='Parameter `a` of beta function', default=2., type=float)
    parser.add_argument('-b', help='Parameter `b` of beta function', default=5., type=float)
    parser.add_argument('-tmin', help='Minimum time to wait (defautl: 100ms)', default=0.1, type=float)
    parser.add_argument('-tmax', help='Maximum time to wait (default: 3s)', default=3.0, type=float)

    return parser.parse_args()


class Sleeper:
    def __init__(self, a: float, b: float, t_min: int, t_max: int, flag: bool=True) -> None:
        """
        :param a:
            Parameter a for beta function.
        :param b:
            Parameter b for beta function.
        :param t_min:
            Minimum time to wait in seconds (fractions of seconds mean milliseconds).
        :param t_max:
            Maximum time to wait in seconds (fractions of seconds mean milliseconds).
        :param flag:
            If false, the wait is not applied.
        """
        self.a = a
        self.b = b
        self.t_min = t_min
        self.t_max = t_max
        self.flag = flag
    
    def active(self, flag: bool) -> None:
        self.flag = flag

    def __call__(self, r: np.random.Generator, thread: int=0):
        """Add some delay in the system.
        
        :param r:
            Random generator.
        :param thread:
            Index of the current thread.
        """
        if self.flag:
            time =  self.t_min + r.beta(self.a, self.b) * (self.t_max - self.t_min)

            logging.info(f'{thread:02} Sleeping for {time*1000:.0f}ms')
            sleep(time)


class TrafficGenerator(multiprocessing.Process):

    def __init__(self, seed: int, config: str, url: str, thread: int, a: float, b: float, t_min: float, t_max: float, event: multiprocessing.Event) -> None:
        """Perform the traffic generation for one worker.
        
        :param N:
            Number of request to generate.
        :param seed:
            Random seed to use.
        :param config:
            Location of the user configuration file.
        :param url:
            The requests will be sent to this endpoint.
        :param thread:
            Number of this thread (for logging purposes).
        :param a:
            Parameter a for beta distribution (used for delays).
        :param b:
            Parameter b for beta distribution (used for delays).
        :param t_min:
            Minimum time to consider for delay (in seconds).
        :param t_max:
            Maximum time to consider for delay (in seconds).
        """
        super().__init__()

        self.random = np.random.default_rng(seed=seed)
        self.user_configs = read_user_config(config)
        self.ul = UserLabeller()
        self.sleep = Sleeper(a, b, t_min, t_max)
        self.thread = thread
        self.url = url
        self.event = event

    def inference_start(self, user: UserData) -> str:
        data = user.dict()
        data['time_arrival'] = str(data['time_arrival'])
        response = requests.post(
            url=f'{self.url}/inference/start', 
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json',
            },
            json=data,
        )

        if response.status_code != 200:
            raise ValueError(f'Inference start: {response.status_code}')

        return response.json()['task_id']


    def inference_status(self, task_id: str) -> bool:
        response = requests.get(
            url=f'{self.url}/inference/status/{task_id}',
            headers={
                'accept': 'application/json',
            },
        )

        if response.status_code != 200:
            raise ValueError(f'Inference status: {response.status_code}')

        status = response.json()
        return status['status']


    def inference_results(self, task_id: str) -> dict:
        response = requests.get(
            url=f'{self.url}/inference/results/{task_id}',
            headers={
                'accept': 'application/json',
            },
        )

        if response.status_code != 200:
            raise ValueError(f'Inference results: {response.status_code}')

        data = response.json()
        return [LocationData(**d) for d in data]


    def make_choice(self, task_id: str, location_id: int) -> dict:
        u_result = requests.put(
            url=f'{self.url}/inference/select/',
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json',
            },
            json={
                'task_id': task_id,
                'location_id': location_id,
            },
        )

        return u_result.json()


    def run(self) -> None:
        thread = self.thread
        r = self.random

        while not self.event.is_set():
            try:
                # choose next user ----------------------------------------
                user_config = r.choice(self.user_configs)

                logging.info(f'{thread:02} User: {user_config["name"]}')

                self.sleep(r, thread)

                user = generate_user_data_from_config(r, user_config)

                # send new inference request ------------------------------
                task_id = self.inference_start(user)

                logging.info(f'{thread:02} Task id assigned: {task_id}')

                # send task get request -----------------------------------
                done = False
                while not done:
                    self.sleep(r, thread)
                    status = self.inference_status(task_id)

                    logging.info(f'{thread:02} Request status: {status}')
                    done = status == 'SUCCESS'

                locations = self.inference_results(task_id)

                # Make choice ---------------------------------------------
                labels = self.ul(r, user, locations)
                location_ids = np.array([l.location_id for l in locations])

                location_id = int(r.choice(location_ids[labels == 1]))
                
                logging.info(f'{thread:02} Choosen location with id {location_id}')

                # Register choice -----------------------------------------
                self.sleep(r, thread)
                self.make_choice(task_id, location_id)

            except ValueError as e:
                logging.error(f'Request failed: {e}')
                self.sleep(r, thread)

        logging.info(f'{self.thread:02} completed')


if __name__ == '__main__':
    from dotenv import load_dotenv

    # Environment variables are controlled through a .env file
    load_dotenv()

    if 'URL' in os.environ:
        URL = os.environ.get('URL')
    else:
        DOMAIN = os.environ.get('DOMAIN', 'localhost')
        URL = f'http://mlpapi.{DOMAIN}'

    args = setup_arguments()

    event = multiprocessing.Event()

    def main_signal_handler(signum, frame):
        if not event.is_set():
            logging.info('Received stop signal')
            event.set()

    signal.signal(signal.SIGINT, main_signal_handler)
    signal.signal(signal.SIGTERM, main_signal_handler)

    n_workers = min(os.cpu_count(), args.p)

    workers = [
        TrafficGenerator(
            args.seed+i, 
            args.config, 
            URL, 
            i+1,
            args.a,
            args.b,
            args.tmin,
            args.tmax,
            event,
        ) for i in range(n_workers)
    ]

    logging.info(f'Starting traffic generation with {n_workers} worker(s)')

    for w in workers:
        w.start()
