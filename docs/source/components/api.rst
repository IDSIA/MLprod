============
API overview
============

In this page we provide a brief overview of our deployment setup while explaining the reasons behind our choices. 

.. note::
    Deployment setups differs from problem to problem, and our proposed solution may not work for a particular use case.
    It is not our idea to impose a particular deployment solution.

The system we want to build has to satisfy the following requirements:

    * We want to develop a service to let the users interact with out model. We use :ref:`FastAPIref` to quickly build a reliable solution.
    * We will have a concurrent setup (many users may interact with the APIs at the same time), so we need a way to schedule tasks in an asynchronous and distributed environment. :ref:`Celeryref` offers us the tools to satisfy those requirements.
    * *Celery* has its own interface: instead of implementing a dedicated piece of code, we use :ref:`RabbitMQref` as message broker.
    * This creates for us an asynchronous environment, where we need a way to cache the results. The NoSQL database :ref:`Redisref` let us make a fast and powerful caching service.
    * Finally, we need a tool to store structured data, like data for users and locations. `PostgreSQL`_ is a modern relational database that does this work.


Basic deployment schema
=======================

.. image:: ../_static/images/API.png
    :align: center

In our setup:

    * The user creates a request through a standard interface provided by FastAPI. This request will be then processed and sent to the message broker RabbitMQ.
    * RabbitMQ will receive the message, process it, and forward it to Celery so it can understand what has to be executed in a dedicated *worker*, and what are the parameters to use.
    * The task is executed in a *worker*. If instructed, it can also access the database for get more data or store partial results.
    * Once the task is finished, Celery retrieves the result and store it in a cache system like Redis.
    * FastAPI pulls the results from the cache and show it to the user.

.. note::

    Our deployment was built as an advanced toy example to show how many components are needed and how they should interact to deploy a system.
    Things like security, hardware scalability, extensive testing, CI, etc... are not considered but are necessary for a real world application.

.. _FastAPIref:

FastAPI
=======

`FastAPI`_ is a open source Python web framework for building web APIs.

With FastAPI we define how the user interact with our application.
The "URL" used by the users are called *routes*.

We implemented routes for:

    * submitting an inference task,
    * check the status of a task,
    * retrieving the predictions,
    * choose the preferred place among the predicted ones,
    * training a new model,
    * get some statistics for monitoring the system.

More specifically, we implemented the following API routes:

.. csv-table::
    :header: "Route", "Description"

    "/inference/start", "Start a new inference task with the current model"
    "/inference/status/{task_id}", "Retrieve the state of the inference task started"
    "/inference/results/{task_id}", "Get the prediction from the model once the inference task is finished"
    "/inference/select/", "Select the best place among the top ranked places"
    "/train/start", "Train a new model"
    "/content/{value}", "Get debugging content from database"
    "/metrics", "Get monitoring statistics"

As an example of how the routes are implemented, the following snippet shows the
code for route */inference/start*. This route consumes the data of a user and it schedules an inference task on celery.

.. literalinclude:: ../../../api/routes.py
    :language: python
    :lines: 50-64

In words, this route first creates a log event, for monitoring and tracking requirements, then create stores the received data in the database and creates an inference task. This task has only the freshly created *user_id* as parameter, since it will collect all the required data during the task itself from the database. Once the inference is started, the resulting *AsyncResult* object is stored in the database, once again for tracking purposes.

Description of every single route is writter in :doc:`routes`.

.. _Celeryref:

Celery
======

`Celery`_ is a distributed task queue that process messages and provides the tools for task scheduling.

We use Celery to perform some computational intensive tasks in an asynchronous way. It is very important to consider that providing just a web application to a user is not enough: if the user starts a heavy task, like our inference, it could be quite annoying to have the website freezed for until the task ends.

Celery needs a startup file and a configuration file. Writing them could be quite challenging for complex system.
In out example we kept them easy and configure just the bare minimum to make it work as intended.

Deploying a system that can scale up with the computational load is mandatory. We implemented these two tasks in Celery:

    * model training,
    * inference on the trained model.

Implementation of these two tasks is described in :doc:`tasks`

Configuration
-------------

*Celery* is configured by using two python files:

    * *celery.py*
    * *celeryconfig.py*

In the first file we create a Python object ``worker`` pointing to a *Celery* instance. 
This object represents the *Celery application* and works as entry point for every operations with Celery.
The object *Celery* is instantiated giving a name, a reference to a backend (in our case :ref:`Redisref`) and a reference to a message broker (in our case :ref:`RabbitMQref`). 
Once the object is created, we load our custom configuration file using ``config_from_object`` and then, at line 17, we start *Celery* itself.

.. literalinclude:: ../../../worker/celery.py
    :linenos:
    :language: python
    :caption: celery.py

The environments variable used in the latter file have to be set at system level (as system environment variables or, in our case, as environment variable in the docker-compose file).

The values we used are:

.. code-block:: python

    CELERY_BROKER_URL=pyamqp://rabbitmq/
    CELERY_BACKEND_URL=redis://redis/

In the file *celeryconfig.py* we configure *Celery* such that it can load and start on request the tasks we want to run asynchronously.

.. literalinclude:: ../../../worker/celeryconfig.py
    :language: python
    :caption: celeryconfig.py

.. _Redisref:

Redis
=====

`Redis`_ is a NoSQL in-memory data store that we use as a cache to temporarily store results from the asynchronous tasks.

In our deployment setup this NoSQL database is only used by FastAPI and Celery to exchange data once the task have been completed.
Since both Celery and FastAPI has a very good compatibility with Redis, the solution is basically plug & play and it does not need configuration.

.. _RabbitMQref:

RabbitMQ
========

`RabbitMQ`_ is a message broker that we use as middleware to send messages from FastAPI to Celery. 

A message broker is usually necessary when two modules needs a translation layer to communicate correctly.
RabbitMQ is a well supported solution which provides a web-based graphical interface useful to monitoring the inter-communication between services.

RabbitMQ has a lot of customizations, but we run it in very basic setup for our toy example, so we can use the default settings.


.. _FastAPI: https://fastapi.tiangolo.com/
.. _Celery: https://docs.celeryq.dev/en/stable/
.. _Redis: https://redis.io/
.. _RabbitMQ: https://www.rabbitmq.com/
.. _PostgreSQL: https://www.postgresql.org/
