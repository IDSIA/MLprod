.. _Dockerref:

======
Docker
======

`Docker`_ is a very powerful container manager implementing the standards provided by the `Cloud Native Computing Foundation`_.
Docker gives to the user a simple interface for creating and managing containers, and it is the *de-facto* standard with a lot of documentation and
a very big community.

.. note::
    Docker installation is possible on Linux, Windows and Mac OS, please refer to the official documentation a https://docs.docker.com.

Our dockers 
===========

In our deployment solution we use 7 dockers interconnected by shared network. These are:

    * FastAPI (REST API)
    * Celery (distributed queue)
    * Redis (in-memory cache)
    * PostgreSQL (relational database)
    * RabbitMQ (message broker)
    * Prometheus (monitoring system)
    * Grafana (statistics dashboard)

Actually there is an additional docker running Traefik which make possible to use sub-domains in out application.

Docker are defined by *Dockerfile* which are text file containing many the commands to setup the image. We use custom
Dockerfiles only for our API and Celery, while for the others software, we use standard docker images pulled from `Docker Hub`_.



**Dockerfile.api**

.. literalinclude:: ../../../Dockerfile.api
    :language: text

**Dockerfile.worker**

.. literalinclude:: ../../../Dockerfile.worker
    :language: text

Docker compose
==============

All the dockers are managed by a *docker-compose* file defined as follows:

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml

Docker compose needs some environmental variables to be set locally before starting the container swarm, this could be
done by creating a *.env* file like the following one:

.. code-block:: text

    DOMAIN=<domain of the machine (used only for traefik labels)>

    CELERY_BROKER_URL=pyamqp://rabbitmq/
    CELERY_BACKEND_URL=redis://redis/
    CELERY_QUEUE=

    DATABASE_SCHEMA=mlpdb
    DATABASE_USER=mlp
    DATABASE_PASS=mlp
    DATABASE_HOST=database

    DATABASE_URL=postgresql://${DATABASE_USER}:${DATABASE_PASS}@${DATABASE_HOST}/${DATABASE_SCHEMA}

    GRAFANA_ADMIN_PASS=grafana



.. _Docker: https://www.docker.com/
.. _Cloud Native Computing Foundation: https://www.cncf.io/
.. _Docker Hub: https://hub.docker.com/