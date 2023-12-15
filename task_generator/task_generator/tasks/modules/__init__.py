from task_generator.constants import Constants
from task_generator.tasks import Task

class TM_Module:

    _TASK: Task

    def __init__(self, task: Task, **kwargs):
        self._TASK = task

    def before_reset(self):
        ...

    def after_reset(self):
        ...