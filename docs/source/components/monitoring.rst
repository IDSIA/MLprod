
==========
Monitoring
==========

*Monitoring* means keeping track of what is happing on your deployment system.
Having an optimal monitoring system helps in:

    * Detecting the problems as soon as they appear
    * Knowing exactly the causes of the problems
    * Knowing where problems happen and the actions to take

A monitring system is usually made by a piece of software that records some metrics and a dashboard
that shows those metrics and statistics. Metrics could be recorded from the infrastructure, like hardware status,
or directly from the code.

To record metrics from the code you usually need additional work, for example, a middleware or some additional
instruction that store somewhere the metrics.

In our project we use :ref:`prometheusref` as middleware to record internal statistics, like KPI and connection, 
and :ref:`grafanaref` to build custom dashboards.

.. _prometheusref:

Prometheus
==========

`Prometheus`_ is a tool that collect metrics by monitoring HTTP endpoints of a target.
Prometheus, in a very basic setup, works by running a server that pull the metrics from some targets and send
them to some data visualization software.

With Prometheus, you define a new component in the system, called middleware, that monitor the routes and
build some statistics. This middleware is defined inside the API and make the statistics readable through its server.

To make Prometheus compatible with FastAPI and our deploy setup, we took inspiration from the following GitHub
repository `prometheusrock`_.

In general, the middleware is made by 2 parts:
    * A middleware python object containing:
        * Objects that defines how data is collected
        * A dispatcher that asynchronously record the data
    * A function that serve the data through dedicated HTTP routes

We defines the tracking objects like in the following snippet: 

.. literalinclude:: ../../../api/middleware/metrics.py
    :language: python
    :linenos:
    :lines: 31-35

Here for example we have Counter for the total number of HTTP request, but many other types are avaiable like Histograms.

Once every statistics is associated to an object, we create an asynchronous function named *dispatch*. 
The so called *dispatcher* is responsible to tracking the statistics and make them aviable.

The last component of our middleware is the function that creates some dedicated HTTP routes where monitoring
software, like Grafana, can retrieve and show the data.

The above description is just a vey simple overview of the components. 
The detailed mechanism that make this piece of code compatible with *FastAPI* is not very intuitive and the explanation
goes for beyond the scope of this documentation. If you want more details, please refere to the aforementioned
GitHub repository.

.. _grafanaref:

Grafana
=======

`Grafana`_ is a monitoring system that allow you to visualize metrics independently from where they are stored.
Practically speaking, Grafana is just a dashboard that can be configured to read metrics wherever you want, in our case Prometheus.

Usually a visualization tools does not need to be embedded in the code, which means it needs only some configuration files
to work. From Grafana 5.0, we can use active provisioning to define datasources and dashboard using configuration files in `yaml` format.

Provisioning
------------

Provisioning defines the process of setting up an IT infrastructure, in our case, datasources and dashboards.
We defined the following provisioning configuration files:

    * dashboards.yaml
    * datasources.yaml


Since Grafana can read data from more than one datasource, we decided to monitor both Prometheus and PostegreSQL.
The following snippet shows the condiguration needed to make the above working:

.. literalinclude:: ../../../grafana/provisioning/datasources/datasources.yml
    :language: yaml
    :linenos:
    :lines: 4-

Here there are shown many settings, but the most important are the URLs and the credentials. Basically we are telling
Grafana where to look for metrics and how to access the componens. For example at line 8 and 24, we are telling
telling to Grafana the URLs of Prometheus and PostgreSQL, and for the latter also the usename, password and the database
name. All this information are used to setup connections and observe the metrics while being published.

Once the datasources are set, we move to the dashboards. Since we want to customize the UI of our monitoring system, we can use
active provisioning to tell Grafana which layout to load. Here there is *yaml* file containing the information to load our
custom dashboard:

.. literalinclude:: ../../../grafana/provisioning/dashboards/dashboards.yaml
    :language: yaml
    :linenos:
    :lines: 3-

Dashboard
---------

Custom dashboards in Grafana can be defined in two ways:

    * By hand
    * By building them directly into the UI and export the setup

Usually the best option is to build one grafically and then export the *json* file, which is quite huge.

A dashboard is composed by panels, and each panel shows a statistic in a specific format.
What a panel does is usually a query to a data source, like Prometheus, and then display the data following
some style directives like, for exmple, the type of plot. 

The following image shows an example of a panel displaying ...

.. .. image:: ../_static/images/grafana.png
..     :align: center



.. _Grafana: https://grafana.com/
.. _Prometheus: https://prometheus.io/
.. _prometheusrock: https://github.com/kozhushman/prometheusrock