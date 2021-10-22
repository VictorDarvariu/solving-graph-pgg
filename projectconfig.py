import os
from math import ceil


class BaseConfig():
    MONGODB_DATABASE_NAME = "graph_gnn_jobs"

    CELERY_TASK_SERIALIZER = "pickle"
    CELERY_RESULT_SERIALIZER = "pickle"
    CELERY_TASK_ACKS_LATE = True
    CELERYD_PREFETCH_MULTIPLIER = 1
    BROKER_POOL_LIMIT = None
    BROKER_HEARTBEAT = 3600.0

class ProjectConfig(BaseConfig):
    BACKEND_URL = f"mongodb://relnet-mongodb:27017/{BaseConfig.MONGODB_DATABASE_NAME}"

    laborer_pw = os.environ["RN_LABORER_PW"]
    CELERY_BROKER_URL = f"pyamqp://relnetlaborer:{laborer_pw}@localhost/relnetvhost"

    NUMBER_WORKER_THREADS = {
        "relnet-worker-cpu": '16',
        "relnet-worker-gpu": '4',
        "relnet-manager": '2'
    }
    WORKER_MAX_TASKS_PER_CHILD = 1

    def get_number_worker_threads(self):
        hostname = os.environ["HOSTNAME"]
        hostname_short = hostname.split(".")[0]
        return self.NUMBER_WORKER_THREADS[hostname_short]

def get_project_config():
    app_settings = ProjectConfig()
    return app_settings