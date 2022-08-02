import requests

from time import sleep
from datetime import date

p_data = requests.post(
    url='http://mlpapi.artemis.idsia.ch/inference/start', 
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

while not done:
    sleep(1)
    g_status = requests.get(
        url=f'http://mlpapi.artemis.idsia.ch/inference/status/{task_id}',
        headers={
            'accept': 'application/json',
        },
    )

    status = g_status.json()
    print(status)
    done = status['status'] == 'SUCCESS'

g_results = requests.get(
    url=f'http://mlpapi.artemis.idsia.ch/inference/results/{task_id}',
    headers={
        'accept': 'application/json',
    },
)

result = g_results.json()
print(result)
print(len(result))