# %%
import numpy as np
import pandas as pd

from api.requests import LocationData

from datas import generate_location_data, generate_user_data, LOCATIONS, UserLabeller

r = np.random.default_rng(42)

USER_CONFIG = 'config/user.tsv'
LOC_CONFIG = 'config/location.tsv'

#%%

user_conf = pd.read_csv(USER_CONFIG, sep='\t')
loc_conf = pd.read_csv(LOC_CONFIG, sep='\t')

# %% -----------------------------------------------------------------------

user_data = []

# generic user
for _ in range(10):
    user_data.append(generate_user_data(r,
        a=1.0, 
        b=1.0
    ))

df_user = pd.DataFrame([x.dict() for x in user_data])
df_user.to_csv('dataset_users.tsv', index=False, header=True, sep='\t')

# %% -----------------------------------------------------------------------
location_data = []

# Zh area: business, lake, high variance between low and high cost

for _ in range(100):
    location_data.append(generate_location_data(r, 
        coordinates=LOCATIONS[0],
        location_distance_min=0,
        location_distance_max=100,
        threshold_breakfast=0.99,
        threshold_lake=0.99,
        price_min=50,
        price_max=1000,
        threshold_mountain=0.2,
        threshold_sport=0.3,
        leisure_min=0.5,
        service_min=0.5,
        score_min=0.3,
        score_max=0.8,
        a=1.0,
        b=1.0,
    ))

# save all objects to a tab-separated value (TSV) file
df_data = pd.DataFrame([x.dict() for x in location_data])
df_data.to_csv('dataset_locations.tsv', index=False, header=True, sep='\t')

# %% -----------------------------------------------------------------------
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
df_ml.to_csv('dataset_labelled.tsv', index=False, header=True, sep='\t')

# %% -----------------------------------------------------------------------
