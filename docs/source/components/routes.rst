==========
API Routes
==========

In this page we will go through all the routes we implemented with *FastAPI*.

Each route is basically a web URL that accept data in a JSON format, performs some operations 
and then reply with a code and a result.

Data exchange
=============

Before getting into details, we need to first introduce how data is shared between the user and the application.
Following the best practices, data has to be shared in a structured way (JSON for example). FastAPI has some neat way
to do this using a module called `pydantic`_.

*Pydantic* defines the format of the data using Python classes, in this way you can shape the information in a strict format
making it easy to manage. In the following snippet of code, there are the classes we wrote to define the structure
of the data. A brief explanation of what each class defines is written as comment.

An example of a data definition class is shown below, full class implementation can be found in *api/requests.py*.

.. literalinclude:: ../../../api/requests.py
    :language: python
    :linenos:
    :lines: 5-9

Routes
======

Now we do a brief explanation of each route implemented in out API.

/inference/start
----------------

.. literalinclude:: ../../../api/routes.py
    :language: python
    :linenos:
    :lines: 51-63

``/inference/start`` is used to do inference with the current active model.

This route receive as parameters:

    * the user's data which contains an ID and some features;
    * a reference to a database.

First of all, we log the event in the database al line 54, then we store the data.
At line 6 we start the asynchronous process by telling Celery to spawn the task called *inference*. ``user_id`` is
then used to retrieve the data stored in the database at line 5.

At line 8 we insert a new entry in the database that help the user tracking the status of the task. Once the task
is finished running it will retrivere the predictions using another route.

The route, at the end, return the status of the task to the user.

/inference/status/{task_id}
---------------------------

.. literalinclude:: ../../../api/routes.py
    :language: python
    :linenos:
    :lines: 66-96

``/inference/status/{task_id}`` is used to check the status of the task running asynchronously.

This route accept as parameter the *id* associated to the process spawned by celery.

/inference/results/{task_id}
----------------------------

.. literalinclude:: ../../../api/routes.py
    :language: python
    :lines: 98-113

/inference/select/
------------------

.. literalinclude:: ../../../api/routes.py
    :language: python
    :lines: 114-133

/train/start
------------

.. literalinclude:: ../../../api/routes.py
    :language: python
    :lines: 134-148

/content/...
------------

.. literalinclude:: ../../../api/routes.py
    :language: python
    :lines: 149-189


.. _pydantic: https://pydantic-docs.helpmanual.io/