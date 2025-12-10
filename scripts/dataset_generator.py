from mlprod.data import (
    read_user_config,
    generate_user_data,
    read_location_config,
    generate_location_data,
)
from mlprod.api.requests import UserData, LocationData
from mlprod.data.labels import UserLabeller
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

import numpy as np
import os
import pandas as pd


r = np.random.default_rng(42)

CONFIG_DIR = Path("configs")


class Config(BaseSettings):
    """Configure the parameters of the dataset generator."""

    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_ignore_unknown_args=True,
        cli_implicit_flags=True,
        extra="forbid",
    )

    user_configs: Path = CONFIG_DIR / "user.tsv"
    location_configs: Path = CONFIG_DIR / "location.tsv"
    dataset_dir: Path = Path("data")

    n_user: int = 1000
    n_loc_per_user: int = 10
    start_date: str = "2026-01-01"


if __name__ == "__main__":
    c = Config()

    print("Input parameters:\n", c.model_dump_json(indent=4))

    os.makedirs(c.dataset_dir, exist_ok=True)

    start_date = np.datetime64(c.start_date)

    # Read pandas into dicts
    print(f"Loading {c.user_configs}... ", end="")
    user_configs = read_user_config(c.user_configs)
    print("Done!")

    print(f"Loading {c.location_configs}... ", end="")
    loc_configs = read_location_config(c.location_configs)
    print("Done!")

    # -----------------------------------------------------------------------

    print("Generating user data...", end="")

    user_data: list[tuple[UserData, UserLabeller]] = []

    # generic user
    for user in user_configs:
        for s in range(user.meta_n):
            user_data.append(
                (
                    generate_user_data(r, user, start_date),
                    UserLabeller(user),
                )
            )

    df_user = pd.DataFrame([x.model_dump() for x, _ in user_data])
    df_user.to_csv(
        c.dataset_dir / "dataset_users.tsv",
        index=False,
        header=True,
        sep="\t",
    )

    print("Done!", "collected", len(user_data), "user types")

    # -----------------------------------------------------------------------

    print("Generating location data... ", end="")

    location_data: list[LocationData] = []

    # Zh area: business, lake, high variance between low and high cost
    for loc in loc_configs:
        for _ in range(loc.meta_n):
            location_data.append(generate_location_data(r, loc))

    # save all objects to a tab-separated value (TSV) file
    df_data = pd.DataFrame([x.model_dump() for x in location_data])
    df_data.drop("location_id", axis=1, inplace=True)
    df_data.to_csv(
        c.dataset_dir / "dataset_locations.tsv",
        index=False,
        header=True,
        sep="\t",
    )

    print("Done!", "collected", len(location_data), "locations")

    # -----------------------------------------------------------------------

    print("Labelling data... ", end="")

    ml_data = []

    user_indices = r.choice(np.arange(len(user_data)), c.n_user)
    users: list[tuple[UserData, UserLabeller]] = [user_data[i] for i in user_indices]

    for user, ul in users:
        loc_indices = r.choice(np.arange(len(location_data)), c.n_loc_per_user)
        locs: list[LocationData] = [location_data[i] for i in loc_indices]
        scores = ul(r, user, locs)

        for i in range(len(locs)):
            d = user.model_dump() | locs[i].model_dump()
            d["label"] = scores[i]
            ml_data.append(d)

    df_ml = pd.DataFrame(ml_data)
    df_ml.drop("location_id", axis=1, inplace=True)
    df_ml.to_csv(
        c.dataset_dir / "dataset_labelled.tsv",
        index=False,
        header=True,
        sep="\t",
    )

    print("Done!", "created dataset of shape", df_ml.shape)
