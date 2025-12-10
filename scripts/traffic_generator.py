from mlprod.api.requests import LocationData, UserData
from mlprod.data import read_user_config, generate_user_data, UserLabeller, UserConfig

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from time import sleep

import logging
import multiprocessing
import numpy as np
import os
import requests
import signal


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)5s %(message)s",
)


class Config(BaseSettings):
    """Configure the parameters of the traffic generator."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_ignore_unknown_args=True,
        cli_implicit_flags=True,
        extra="forbid",
    )

    """Set fixed seed."""
    seed: int = 42
    """Number of requests to do, 0 means infinite"""
    n: int | None = None
    """If set, no sleeps are used"""
    no_sleep: bool = False
    """User config file to use"""
    config: Path = Path("./config/user.tsv")

    """Decision level, the amount of responses that will also have a feedback (set a label)."""
    d: float = 1.0
    """Number of parallel generators."""
    p: int = 1
    """Parameter `a` of beta function."""
    a: float = 2.0
    """Parameter `b` of beta function."""
    b: float = 5.0
    """Minimum time to wait in seconds."""
    tmin: float = 0.1
    """Maximum time to wait in seconds."""
    tmax: float = 3.0


class TrafficGenerator(multiprocessing.Process):
    """Traffic generator object."""

    def __init__(
        self,
        seed: int,
        config: Path,
        url: str,
        thread: int,
        a: float,
        b: float,
        t_min: float,
        t_max: float,
        decision: float,
        event: multiprocessing.Event,  # type: ignore
        flag: bool = False,
    ) -> None:
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
        :param flag:
            Sleep flag, if True disable all sleeps.
        """
        super().__init__()

        self.random = np.random.default_rng(seed=seed)
        self.user_configs = read_user_config(config)
        self.thread = thread
        self.url = url
        self.event = event
        self.decision = decision
        self.a = a
        self.b = b
        self.t_min = t_min
        self.t_max = t_max
        self.flag = flag

    def sleep(self):
        """Applies a delay to the execution of some part of this script."""
        if self.flag:
            return

        time = self.t_min + self.random.beta(self.a, self.b) * (self.t_max - self.t_min)

        logging.info(f"{self.thread:02} Sleeping for {time * 1000:.0f}ms")
        sleep(time)

    def inference_start(self, user: UserData) -> str:
        """Sends the user data to the application to simulate the search performed by a user and start an inference."""
        data = user.model_dump()
        data["time_arrival"] = str(data["time_arrival"])
        response = requests.post(
            url=f"{self.url}/inference/start",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            json=data,
        )

        if response.status_code != 200:
            raise ValueError(f"Inference start: {response.status_code}")

        return response.json()["task_id"]

    def inference_status(self, task_id: str) -> bool:
        """Contact the application to get information on the status of the inference."""
        response = requests.get(
            url=f"{self.url}/inference/status/{task_id}",
            headers={
                "accept": "application/json",
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Inference status: {response.status_code}")

        status = response.json()
        return status["status"]

    def inference_results(self, task_id: str) -> list[LocationData]:
        """Get the results from the application for an inference that has been completed."""
        response = requests.get(
            url=f"{self.url}/inference/results/{task_id}",
            headers={
                "accept": "application/json",
            },
        )

        if response.status_code != 200:
            raise ValueError(f"Inference results: {response.status_code}")

        data = response.json()
        return [LocationData(**d) for d in data]

    def make_choice(self, task_id: str, location_id: int) -> dict:
        """Simulate the act of a user of choosing a destination and send the chosen location to the application."""
        u_result = requests.put(
            url=f"{self.url}/inference/select/",
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            json={
                "task_id": task_id,
                "location_id": location_id,
            },
        )

        return u_result.json()

    def run(self) -> None:
        """Main loop of the user-simulation.

        While not stopped, continues to create user researches.
        The steps of the simulations are:
        * choose the user to simulate
        * send an inference request
        * polling until the results are ready
        * get the results
        * make and register a choice

        Sleep delays (if enabled) are used to simulate the slowly process of decision by an user.
        """
        thread = self.thread
        r = self.random

        while not self.event.is_set():
            try:
                # choose next user ----------------------------------------
                x = r.choice(np.arange(len(self.user_configs)))
                user_config: UserConfig = self.user_configs[x]

                logging.info(f"{thread:02} User: {user_config.meta_comment}")

                self.sleep()

                start_date = np.datetime64("2026-01-01")

                user = generate_user_data(r, user_config, start_date)
                ul = UserLabeller(user_config)

                # send new inference request ------------------------------
                task_id = self.inference_start(user)

                logging.info(f"{thread:02} Task id assigned: {task_id}")

                # send task get request -----------------------------------
                done = False
                while not done:
                    self.sleep()
                    status = self.inference_status(task_id)

                    logging.info(f"{thread:02} Request status: {status}")
                    done = status == "SUCCESS"

                locations = self.inference_results(task_id)
                location_id = -1
                # Make choice ---------------------------------------------
                if r.uniform() < self.decision:
                    labels = ul(r, user, locations)

                    if labels.sum() > 0:
                        location_ids = np.array(
                            [location.location_id for location in locations]
                        )
                        location_id = int(r.choice(location_ids[labels == 1]))

                    logging.info(f"{thread:02} Chosen location with id {location_id}")

                # Register choice -----------------------------------------
                self.sleep()
                self.make_choice(task_id, location_id)

            except ValueError as e:
                logging.error(f"Request failed: {e}")
                self.sleep()

        logging.info(f"{self.thread:02} completed")


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Environment variables are controlled through a .env file
    load_dotenv()

    if "URL" in os.environ:
        URL = os.environ.get("URL", "")
    else:
        DOMAIN = os.environ.get("DOMAIN", "localhost")
        URL = f"http://mlpapi.{DOMAIN}"

    config = Config()

    event = multiprocessing.Event()

    def main_signal_handler(signum, frame):
        """This handler is used to gracefully stop the threads when ctrl-c is hitted in the terminal."""
        if not event.is_set():
            logging.info("Received stop signal")
            event.set()

    signal.signal(signal.SIGINT, main_signal_handler)
    signal.signal(signal.SIGTERM, main_signal_handler)

    n_workers = min(os.cpu_count() or 1, config.p)

    workers = [
        TrafficGenerator(
            config.seed + i,
            config.config,
            URL,
            i + 1,
            config.a,
            config.b,
            config.tmin,
            max(config.tmin, config.tmax),
            config.d,
            event,
        )
        for i in range(n_workers)
    ]

    logging.info(f"Starting traffic generation with {n_workers} worker(s)")

    for w in workers:
        w.start()
