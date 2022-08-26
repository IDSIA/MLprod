# Grafana storage folder

[Grafana](https://grafana.com/) is a powerful tool used to build dashboards and interfaces for monitoring systems. We are using this tool to build a simple dashboard and check the data stored in the `database` and the metrics collected with the `prometheus` service.

## Folders description

Files under the `provisioning` folder are used to define the *data sources* and *dashboards* available at the start of the application.

Files under the `dashboards` folder are the dashboards that will be visualized in Grafana.

Remember that the dashboards uploaded in this way are not editable.
