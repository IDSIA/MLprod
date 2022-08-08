# %%
import numpy as np
import pandas as pd

from datas import read_user_config, generate_user_data_from_config, read_location_config, generate_location_data_from_config, UserLabeller
from datas.users import generate_user_labeller_from_config

r = np.random.default_rng(42)

USER_CONFIG = 'config/user.tsv'
LOC_CONFIG = 'config/location.tsv'

DATASET_USER = 'dataset/dataset_users.tsv'
DATASET_LOC = 'dataset/dataset_locations.tsv'
DATASET_LABEL = 'dataset/dataset_labelled.tsv'

#%%

# Read pandas into dicts
print(f"Loading {USER_CONFIG}... ", end="")
user_settings = read_user_config(USER_CONFIG)
print("Done")

print(f"Loading {LOC_CONFIG}... ", end="")
loc_settings = read_location_config(LOC_CONFIG)
print("Done")

# %% -----------------------------------------------------------------------

print("Generating user data...", end="")

user_data = []

# generic user
for user in user_settings:
    for s in range(user['qnt']):
        user_data.append((
            generate_user_data_from_config(r, user),
            generate_user_labeller_from_config(user),
        ))

df_user = pd.DataFrame([x.dict() for x in user_data])
df_user.to_csv(DATASET_USER, index=False, header=True, sep='\t')

print("Done")

# %% -----------------------------------------------------------------------

print("Generating location data... ", end="")

location_data = []

# Zh area: business, lake, high variance between low and high cost
for loc in loc_settings:
    for _ in range(loc['qnt']):
        location_data.append(
            generate_location_data_from_config(r, loc)
        )

# save all objects to a tab-separated value (TSV) file
df_data = pd.DataFrame([x.dict() for x in location_data])
df_data.to_csv(DATASET_LOC, index=False, header=True, sep='\t')

print("Done")

# %% -----------------------------------------------------------------------

print("Labelling data... ", end="")

ml_data = []
users = r.choice(user_data, 1000).tolist()

for user, ul in users:
    locs = r.choice(location_data, 10).tolist()
    scores = ul(r, user, locs)

    for i in range(len(locs)):
        d = user.dict() | locs[i].dict()
        d['label'] = scores[i]
        ml_data.append(d)

df_ml = pd.DataFrame(ml_data)
df_ml.to_csv(DATASET_LABEL, index=False, header=True, sep='\t')

print("Done")
