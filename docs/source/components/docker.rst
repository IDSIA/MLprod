.. _Dockerref:

======
Docker
======

`Docker`_ is a very powerful container manager implementing the standards provided by the `Cloud Native Computing Foundation`_.
Docker gives to the user a simple interface for creating and managing containers, and it is the *de-facto* standard with a lot of documentation and
a very big community.

.. note::
    Docker installation is possible on Linux, Windows and Mac OS, please refer to the official documentation a https://docs.docker.com.

In this page we will explain how we made every components of our deployment system running inside a container.
We will describe our custom-made Dockerfiles and how we manage to interconnect every container using *docker-compose*.

Introduction
============

What is a container?

    *A container is a standard unit of software that packages up code and all its dependencies so the application runs quickly and reliably from one computing environment to another. A Docker container image is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, system libraries and settings.* [`Source`_]

Every single container contains all the required code needed to setup and run an application.
This includes, but it is not mandatory to have, a minimal operating system.
The *Dockerfile* defines the commands and operations needed to setup the environment for the application. 

Once a *Dockerfile* is written, it needs to be "built" into an *image* and images are nothing more than a snapshot of the initial state of a container.
Starting a container consist of taking an image and execute what is inside.

Independent container can communicate between each other through what is called *a docker network*, which can be seen as a virtual local network created and managed by the Docker daemon.
Many containers can be spawned spawned at the same time and connected to the same docker network, creating a distributed system.

Docker offers two way to manage group of container that need to interact in a virtual network: ``docker-compose`` and ``Docker Swarm``. The first is a tool that simplify the process of writing and execute docker commands in a structured way; the second is a product developed by Docker to use a similar structure of files to build and manage services is a cluster of machines.

In our deployment system, to make our application running in one command, we use a simple ``docker-compose``. This tool lets you setup the entire stack of services in a single yaml file. Each container has it own entry in the file, and the developer can define additional settings like environment variable.

Our custom dockers 
==================

In our deployment solution we use 7 dockers interconnected by a shared network. These are:

    * FastAPI (REST API)
    * Celery (distributed queue)
    * Redis (in-memory cache)
    * PostgreSQL (relational database)
    * RabbitMQ (message broker)
    * Prometheus (monitoring system)
    * Grafana (statistics dashboard)

Actually there is an additional docker running `Traefik`_ which make possible to use sub-domains in our application. This service is external to our application and it is not covered by this documentation.

Docker containers are defined by *Dockerfile* which are text file containing the commands to setup the image.
We use custom Dockerfiles only for our API and Celery, while for the others software, we use public available docker images pulled from `Docker Hub`_.

.. Note::
    Our Dockerfiles may contains some optimization, like multi-stage build, that is out of the scope of this documentation.
    Please refer to the official docker documentation for additional material if interested.

FastAPI
-------

Running *FastAPI* requires a customized docker image since we need a linux OS with python and other tools installed.
In **Dockerfile.api** we defined the environment for our application and it is shown below with line-by-line explanations.

* We tell Docker to start from a generic image containing Python 3.10. This image is used to build a *virtual environment* with all the required dependencies to run our api application.

.. literalinclude:: ../../../Dockerfile.api
    :language: docker
    :lines: 3-4

* We create a *virtual env* environment and install the python packages from the file requirements.txt (copied from the local storage)

.. literalinclude:: ../../../Dockerfile.api
    :language: docker
    :lines: 6-14

* We exploit the multi-stage build for better performances and less memory consumption by adding a second stage from a smaller image.

.. literalinclude:: ../../../Dockerfile.api
    :language: docker
    :lines: 16-17

* We add python from the virtual environment to the linux PATH environment variable. This is a trick to enable the python *virtual environment*.

.. literalinclude:: ../../../Dockerfile.api
    :language: docker
    :lines: 18

* We copy the virtual environment from the previous build stage to the current one.

.. literalinclude:: ../../../Dockerfile.api
    :language: docker
    :lines: 20-21

* We setup the working directory and copy from the local storage the necessary files.

.. literalinclude:: ../../../Dockerfile.api
    :language: docker
    :lines: 23-29

* We setup the command to run *FastAPI* when the docker image is spawned inside a container. 

.. literalinclude:: ../../../Dockerfile.api
    :language: docker
    :lines: 31


Celery worker
-------------

Running *Celery* requires a customized docker image just like *FastAPI*. 
In **Dockerfile.worker** we define the environment for the distributed queue manager. Line-by-line explanation is done below.

* We perform the same operations explained for *FastAPI*. The only change is in the python requirements used to setup the virtual environment.

.. literalinclude:: ../../../Dockerfile.worker
    :language: docker
    :lines: 3-21

* We setup the working directory and copy from the local storage the necessary files. We copy the tasks to be run asynchronously and the database classes.

.. literalinclude:: ../../../Dockerfile.worker
    :language: docker
    :lines: 23-26

* We setup the command to run *Celery* when the docker image is spawned inside a container. 

.. literalinclude:: ../../../Dockerfile.worker
    :language: docker
    :lines: 28


Docker compose
==============

Our custom dockers and all the others are managed by *docker-compose.yml*.

We will now go through the definition of each service, starting first from the internal network definition that makes possible for them to communicate internally.

* The docker-compose ``version`` set the supported version of the configuration file for our docker-compose tool.

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml
    :lines: 1


* Internal network definition, *mlpnet* and *www* are the aliases for the 2 networks. While *mlpnet* is created and managed by the docker-compose, the *www* network is defined externally. This means both that this network need to exists *before* the start our dockers and that it will still exist *after* we stop our dockers. The *www* network is used to expose externally our dockers via Traefik.

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml
    :lines: 3-8

* We run RabbitMQ without any particular configuration. We just start the container using default configuration. Ports are always open inside a docker network, so we don't need to expose explicitly the communication port 5672 over the *mlpnet*: define the network to use is enough.

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml
    :lines: 12-17

* Also Redis is run using default settings an the *mlpnet* network. Like RabbitMQ, the communication port 5672 is already available.

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml
    :lines: 19-24

* Prometheus needs a little bit of configuration. We start from a standard image and give access to both *mlpnet* and *www* networks.

  Prometheus configuration file is loaded by telling docker-compose to mount the *prometheus.yaml* file inside the docker. *Labels* tags are for Traefik only.

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml
    :lines: 25-37

* PostgreSQL only needs some environment variables, so we start from a standard docker image and specify the variables using tag *environment:*
  We then set the network to *mlpnet*. Once again, port 5432 is open by default.

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml
    :lines: 39-48

* FastAPI and Celery have a similar setup, both have a custom Dockerfile to be built and some dependencies to be installed.

  We tell docker-compose to build a local image with the tag *build:* and by indicating the dedicated Dockerfile to use.

  Environment variable are necessary to set setup parameters for Celery and communicate with the DB. Note how it is possible to use variables (`${something}`) inside the declaration of a environment variable for Docker. These external variables can be set in the command line or inside a file named `.env`. More on this in the next section.
  
  Using *volumes:* we can map the local model folder inside the docker to make the model accessible on both sides.
  
  In the end, we define the network and the dependencies to be satisfied to start the container, those are *Redis*, *RabbitMQ* and *PostgreSQL* up and running. 

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml
    :lines: 50-91

* Grafana is the last container we have in our system, it provides a customized dashboard where we can track any metrics we want. For more information about how we monitor the entire deploy please refer to :doc:`monitoring` page.

  With Grafana, we start from a standard docker image, but since monitoring requires reading information from various source we need to 
  set some environment variables to tell Grafana where and how to access Prometheus and PostgreSQL.

  We mount as volumes the customized dashboard and provisioning folder and connect the container to both *mlpnet* and *www* networks.
  Since Grafana needs PostgreSQL and Prometheus to work, we set those two containers as dependencies.

.. literalinclude:: ../../../docker-compose.yaml
    :language: yaml
    :lines: 93-


Environment variable
====================

Docker compose needs some environmental variables to be set locally before starting the container stack. 
This could be done by creating a ``.env`` file with the following content:

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

In our case, environment variables are nothing more than URL strings and login credentials.

.. _Docker: https://www.docker.com/
.. _Cloud Native Computing Foundation: https://www.cncf.io/
.. _Docker Hub: https://hub.docker.com/
.. _Traefik: https://traefik.io/
.. _Source: [https://www.docker.com/resources/what-container/]