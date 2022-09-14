
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
repository `prometheusrock`_. We will now go through the main parts of the code and explain how they work.

.. _grafanaref:

Grafana
=======

`Grafana`_ is a monitoring system that allow you to visualize metrics independently from where they are stored.
Practically speaking, Grafana is just a dashboard that can be configured to read metrics wherever you want, in our case Prometheus.

Usually a visualization tools does not need to be embedded in the code, which means it needs only some configuration files
to work. With Grafana we need to define the following configuration files in `yaml` and `json` format:

    * dashboards.yaml
    * datasources.yaml
    * monitoring.json

Data sources
------------

Since Grafana can read data from more than one datasource, we decided to monitor both Prometheus and PostegreSQL.
The following snippet shows the condiguration needed to make the above working:

.. literalinclude:: ../../../grafana/provisioning/datasources/datasources.yml
    :language: yaml
    :linenos:
    :lines: 5-

Here there are shown many settings, but the most important are the URLs and the credentials. Basically we are telling
Grafana where to look for metrics and how to access the componens. For example at line 8 and 24, we are telling
telling to Grafana the URLs of Prometheus and PostgreSQL, and for the latter also the usename, password and the database
name. All this information are used to setup connections and observe the metrics while being published.



.. _Grafana: https://grafana.com/
.. _Prometheus: https://prometheus.io/
.. _prometheusrock: https://github.com/kozhushman/prometheusrock