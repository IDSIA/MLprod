# Machine Learning in Production 


# Run the dockers

To run the dockers, first create a `.env` file with the following content:

```
DOMAIN=<domain of the machine (used only for traefik labels)>

CELERY_BROKER_URL=pyamqp://rabbitmq/
CELERY_BACKEND_URL=redis://redis/
CELERY_QUEUE=

DATABASE_SCHEMA=mlpdb
DATABASE_USER=mlp
DATABASE_PASS=mlp
DATABASE_HOST=database
```

Then launch the docker through the docker compose:

```
docker-compose up -d
```
