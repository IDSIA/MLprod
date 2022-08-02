"""
NOTE: launch this script with something like

    URL=http://.../ python debug_requests.py
"""

import requests
import os

from time import sleep
from datetime import date


URL = os.environ.get('URL', 'localhost')

# start a new inference
p_data = requests.post(
    url=f'{URL}/inference/start', 
    headers={
        'accept': 'application/json',
        'Content-Type': 'application/json',
    },
    json={
        'people_num': 2,
        'people_age': [32, 35],
        'children': 0,
        'budget': 3000,
        'nights': 5,
        'time_arrival': str(date.today()),
        'pool': True,
        'spa': True,
        'pet_friendly': False,
        'lake': True,
        'mountain': True,
        'sport': False,
    },
)

data = p_data.json()
print(data)

task_id = data['task_id']

done = False

# wait for answer
while not done:
    sleep(1)
    g_status = requests.get(
        url=f'{URL}/inference/status/{task_id}',
        headers={
            'accept': 'application/json',
        },
    )

    status = g_status.json()
    print(status)
    done = status['status'] == 'SUCCESS'

# get results
g_results = requests.get(
    url=f'{URL}/inference/results/{task_id}',
    headers={
        'accept': 'application/json',
    },
)

result = g_results.json()

print(len(result))
for r in result:
    print(r['location_id'], r['score'])

# select first (higher score) as label
location_id = result[0]['location_id']
print('selected location_id =', location_id)

u_result = requests.put(
    url=f'{URL}/select/',
    headers={
        'accept': 'application/json',
        'Content-Type': 'application/json',
    },
    json={
        'task_id': task_id,
        'location_id': location_id,
    },
)

res = u_result.json()
print(res)

res_id = res['id']

# check that the update was executed correctly
g_result = requests.get(
    url=f'{URL}/content/result/{res_id}',
    headers={
        'accept': 'application/json',
    },
    data={
        'location_id': location_id
    },
)

res_data = g_result.json()
print(res_data)
