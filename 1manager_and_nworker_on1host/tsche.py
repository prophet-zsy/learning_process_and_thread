# taskschedule
import os, sys, queue, time, random, re, json
import datetime, traceback, pickle
import multiprocessing, copy, signal

# signal.signal(signal.SIGCHLD,signal.SIG_IGN)

RESOURE_NUM = 5  # max num of tasks running at the same time


class ScheduleItem:
    def __init__(self, item, task_id):
        # task content (assign in initial)
        self.item = item
        self.task_id = task_id

        # task info
        self.sche_id = -1  # assign in TaskScheduler().exec_task_async
        self.pid = -1  # assgin in task_func
        self.start_time = None # assign in task_func
        self.cost_time = 0 # assign in task_func
        self.resource_id = -1  # ScheduleItem give resource to it

        # result
        self.result = None # assign in task_func

class TaskScheduler:
    #  Mainly for coordinating resources in multi workers
    def __init__(self):
        self.task_list = []
        self.result_list = []

        # for multiprocessing communication
        self.result_buffer = multiprocessing.Queue()
        self.signal = multiprocessing.Event()
        
        # store all the running sub process object, and we will use it when make sure the subprocess return
        self.sub_process = {}

        # record the info of every task and errinfo
        self.log_path = os.path.join(os.getcwd(), "log")
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        
        # resource
        self.resource_num = RESOURE_NUM
        self.resource_list = queue.Queue()
        for resource in range(self.resource_num):
            self.resource_list.put(resource)
        
        # for counting task(every task has a unique sche_id)
        self.sche_cnt = 0

    def load_tasks(self, tasks):
        self.task_list.extend(tasks)

    def get_sche_id(self):
        tmp_id = self.sche_cnt
        self.sche_cnt += 1
        return tmp_id

    def all_alive(self):
        for subpid in self.sub_process.keys():
            if not self.sub_process[subpid].is_alive:
                return False
        return True

    def makesure_sub_return(self, subpid):
        # make sure the sub zombie process vanish
        # pid, exit_code = os.waitpid(subpid, 0)
        # p_wait_vanish = self.sub_process[subpid]
        # p_wait_vanish.join(timeout=10)  # block itself to wait for the subprocess's return
        # if p_wait_vanish.is_alive:
        #     p_wait_vanish.terminate()  # stop it forcely
        #     p_wait_vanish.join()
        del self.sub_process[subpid]

    def exec_task_async(self, task_func, *args, **kwargs):
        """Async: directly return whetherever the tasks is completed
        """
        while self.task_list and not self.resource_list.empty():
            resource = self.resource_list.get()  # get resource
            task_item = self.task_list.pop(0)  # get task
            # config task
            task_item.resource_id = resource
            task_item.sche_id = self.get_sche_id()
            # exec task
            subp = multiprocessing.Process(target=task_fun_parent, args=[task_func, task_item, self.result_buffer, self.signal, *args])
            subp.start()
            # sign up the subpid
            self.sub_process[subp.pid] = subp
        self.signal.clear()

    def load_part_result(self):
        """load one or more results if there are tasks completed
        """
        # while self.all_alive():  # when all the subp is alive
        #     self.signal.wait(timeout=100)  # check the subp every 100s
        self.signal.wait()
        # while not self.result_buffer.empty():  # empty() is unreliable!!!
        # while self.result_buffer.qsize() > 0:  # qsize() is unreliable!!!
        while True:
            try:
                task_item = self.result_buffer.get(timeout=2)
            except:
                break
            self.result_list.append(task_item)
            self.resource_list.put(task_item.resource_id)  # return resource
            self.makesure_sub_return(task_item.pid)

    def exec_task(self, task_func, *args, **kwargs):
        """Sync: waiting for all the tasks completed before return
        """
        while self.task_list or self.resource_list.qsize() < self.resource_num:
            self.exec_task_async(task_func, *args, **kwargs)
            self.load_part_result()

    def get_result(self):
        self.dump_task_info()

        result = self.result_list
        self.result_list = []
        return result

    def dump_task_info(self):
        for task_item in self.result_list:
            task_dump_file = "scheid{}taskid{}.pickle".format(task_item.sche_id, task_item.task_id)
            with open(os.path.join(self.log_path, task_dump_file), "wb") as f:
                pickle.dump(task_item, f)

def err_log(task_item, error):
    with open(os.path.join(os.getcwd(), "err.log"), "a") as f:
        f.write("#########Some errors occur in the task: scheid {} taskid {} #########\n".format(task_item.sche_id, task_item.task_id))
        f.write("pid {} resourceid {} start_time {} \n".format(task_item.pid, task_item.resource_id, task_item.start_time))
        f.write(error)
    
def task_fun_parent(task_func, task_item, result_buffer, signal, *args, **kwargs):
    time_cnt = TimeCnt()
    task_item.pid = os.getpid()
    task_item.start_time = time_cnt.start()

    try:
        task_item = task_func(task_item, *args)
    except Exception as error:
        err_log(task_item, error)
        task_item.result = None 

    task_item.cost_time = time_cnt.stop()
    # record the result
    result_buffer.put(task_item)
    signal.set()

#  for test...
def task_fun(task_item, *args, **kwargs):
    print("{}  using resource {} start_time {}".format(task_item.item, task_item.resource_id, task_item.start_time))

    time.sleep(random.randint(2,20))

    print("{}  using resource {} cost_time {}  #######finished########".format(task_item.item, task_item.resource_id, task_item.cost_time))
    
    return task_item


class TimeCnt:
	def __init__(self):
		self.time_stamp = None

	def start(self):
		self.time_stamp = datetime.datetime.now()
		start_time = self.time_stamp.strftime('%d %H:%M:%S')
		return start_time

	def stop(self):
		cost_time = (datetime.datetime.now() - self.time_stamp)
		# format the costtime
		total_seconds = int(cost_time.total_seconds())
		hours = total_seconds//3600
		seconds_inHour = total_seconds%3600
		minutes = seconds_inHour//60
		seconds = seconds_inHour%60
		cost_time = '{}:{}:{}'.format(hours, minutes, seconds)
		return cost_time

if __name__ == '__main__':
    tasks = []
    for i in range(15):
        tasks.append(ScheduleItem("I am a task {} ".format(i), task_id=i))
    TSche = TaskScheduler()
    TSche.load_tasks(tasks)
    TSche.exec_task(task_fun)
    result = TSche.get_result()

