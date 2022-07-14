# %%
import numpy as np
import pandas as pd

from datas import generate_location_data, generate_user_data, UserLabeller

r = np.random.default_rng(42)

USER_CONFIG = 'config/user.tsv'
LOC_CONFIG = 'config/location.tsv'

DATASET_USER = 'dataset/dataset_users.tsv'
DATASET_LOC = 'dataset/dataset_locations.tsv'
DATASET_LABEL = 'dataset/dataset_labelled.tsv'

#%%

print(f"Loading {USER_CONFIG}... ", end="")
user_conf = pd.read_csv(USER_CONFIG, sep='\t')
print("Done")

print(f"Loading {LOC_CONFIG}... ", end="")
loc_conf = pd.read_csv(LOC_CONFIG, sep='\t')
print("Done")

#%%

# Convert pandas rows in dicts

user_settings = []
for _, row in user_conf.iterrows():
    user_dict = {
        'name': row['meta_comment'],
        'qnt': row['meta_dist'],
        'settings': row.iloc[2:].to_dict()
    }
    user_settings.append(user_dict)

loc_settings = []
for _, row in loc_conf.iterrows():
    loc_dict = {
        'name': row['meta_comment'],
        'qnt': row['meta_n'],
        'settings': row.iloc[2:].to_dict()
    }
    loc_settings.append(loc_dict)

# %% -----------------------------------------------------------------------

print("Generating user data...", end="")

user_data = []

# generic user
for user in user_settings:
    for s in range(user['qnt']):
        user_data.append(
            generate_user_data(r, **user['settings'])
        )

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
            generate_location_data(r, **loc['settings'])
        )

# save all objects to a tab-separated value (TSV) file
df_data = pd.DataFrame([x.dict() for x in location_data])
df_data.to_csv(DATASET_LOC, index=False, header=True, sep='\t')

print("Done")

# %% -----------------------------------------------------------------------

print("Labelling data... ", end="")

ul = UserLabeller()

ml_data = []
users = r.choice(user_data, 100).tolist()

for user in users:
    locs = r.choice(location_data, 10).tolist()
    scores = ul(r, user, locs)

    for i in range(len(locs)):
        d = user.dict() | locs[i].dict()
        d['label'] = scores[i]
        ml_data.append(d)

df_ml = pd.DataFrame(ml_data)
df_ml.to_csv(DATASET_LABEL, index=False, header=True, sep='\t')

print("Done")
