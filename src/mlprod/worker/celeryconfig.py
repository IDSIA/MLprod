result_expires = 3600

# Ignore other content
accept_content = ["json"]
task_serializer = "json"
result_serializer = "json"

timezone = "Europe/Zurich"
enable_utc = True

imports = (
    "mlprod.worker.tasks",
    "mlprod.worker.models",
    "mlprod.database",
    "mlprod.database.crud",
)

include = [
    "mlprod.worker.tasks.inference",
    "mlprod.worker.tasks.train",
]
