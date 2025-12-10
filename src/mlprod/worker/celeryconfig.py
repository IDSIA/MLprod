result_expires = 3600

# Ignore other content
accept_content = ["json"]
task_serializer = "json"
result_serializer = "json"

timezone = "Europe/Zurich"
enable_utc = True

imports = (
    "worker.tasks",
    "worker.models",
    "database",
    "database.crud",
)

include = [
    "worker.tasks.inference",
    "worker.tasks.train",
]
