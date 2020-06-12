import time, random, os, sys, re, copy
import traceback
from tsche import TaskScheduler, ScheduleItem

# do not put the func into any class, 
# because child processes in Python cannot have member functions of any class as entry points
def task_fun(task_item, *args, **kwargs):

    # do something you want to do
    time.sleep(random.randint(2,20))
    task_item.result = None

    return task_item


if __name__ == '__main__':
    tasks = []
    for i in range(15):
        tasks.append(ScheduleItem("I am a task {} ".format(i), task_id=i))
    added_param = None  # if you want to use something in subprocess
    
    TSche = TaskScheduler()
    TSche.load_tasks(tasks)
    TSche.exec_task(task_fun, added_param)
    result = TSche.get_result()



