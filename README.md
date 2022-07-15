# Machine Learning in Production 

## Run the dockers

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

GRAFANA_ADMIN_PASS=grafana
```

Then launch the docker through the docker compose:

```
docker-compose up -d
```

## References

### FastAPI and database interaction
* [SQL (Relational) Databases](https://fastapi.tiangolo.com/tutorial/sql-databases/)
* [Python ML in Production - Part 1: FastAPI + Celery with Docker](https://denisbrogg.hashnode.dev/python-ml-in-production-part-1-fastapi-celery-with-docker)
* [First Steps with Celery](https://docs.celeryq.dev/en/stable/getting-started/first-steps-with-celery.html)
* [Next Steps](https://docs.celeryq.dev/en/stable/getting-started/next-steps.html)
* [Serving ML Models in Production with FastAPI and Celery](https://towardsdatascience.com/deploying-ml-models-in-production-with-fastapi-and-celery-7063e539a5db)
* [Multi-stage builds #2: Python specifics](https://pythonspeed.com/articles/multi-stage-docker-python/#solution2-virtualenv)
* [SQLAlchemy ORM — a more “Pythonic” way of interacting with your database](https://medium.com/dataexplorations/sqlalchemy-orm-a-more-pythonic-way-of-interacting-with-your-database-935b57fd2d4d)
* [Events: startup - shutdown](https://fastapi.tiangolo.com/advanced/events/)

### Metrics with Prometheus
* [Overview | Prometheus](https://prometheus.io/docs/introduction/overview/)
* [Instrumentation | Prometheus](https://prometheus.io/docs/practices/instrumentation/#counter-vs-gauge-summary-vs-histogram)
* [prometheus/client_python | GitHub](https://github.com/prometheus/client_python)
* [kozhushman/prometheusrock | GitHub](https://github.com/kozhushman/prometheusrock)
