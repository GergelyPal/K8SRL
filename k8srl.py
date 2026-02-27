import argparse
import os
import ray
import math
from ray.rllib.env.env_context import EnvContext
#  from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.algorithms import dqn
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import get_trainable_cls
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from typing import Any
#  import gym
from gymnasium.spaces import Tuple, Box, Discrete
import numpy as np
from typing import List
import sys
import time

import simpy
import random

from tdigest import TDigest
from enum import IntEnum, Enum
from collections import namedtuple
import itertools

RANDOM_SEED = 42

BOOT_TIME_NODE = 150
WAKE_TIME_NODE = 1
BOOT_TIME_POD = 50

IDLE_ACTIVE_RATIO = 0.2

SIM_TIME = 110000000            # Simulation time in seconds
NUM_ARRIVALS = 10000
CLUSTER_CONTROL_TIME = 100

NUMBER_OF_NODES = 20
NODE_RAM = 30000
NODE_CORE = 8
POD_USAGE = 90

ARRIVAL_RATE = 0.6
VARIATION = 0

SERVICE_RATE = 0.05



tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="DQN", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=500000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=1000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use DQN without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


class Action(IntEnum):
    ScaleOut = 0
    Do_nothing = 1
    ScaleIn = 2

class PowerState(IntEnum):
    OFF = 0
    ON = 1
    IDLE = 2

#hostokbol lettek a node-ok
#serverekbol lettek a podok

Resources = namedtuple("Resources", ["ram", "cpu"])

class ServiceType(Enum):
    typeA = Resources(ram = 128., cpu = 0.1)
    typeB = Resources(ram = 256., cpu = 0.2)
    typeC = Resources(ram = 512., cpu = 0.4)

class Pod(object):
    def __init__(self, env: Any, pod_id: int, service_type: ServiceType , power_state: PowerState) -> None:
        self.env = env
        self.id = pod_id
        self.birth = env.now
        self.service_type = service_type

        self.power_state = power_state
        self.ram = simpy.Container(self.env, service_type.value.ram, service_type.value.ram)
        self.cpu = simpy.Container(self.env, service_type.value.cpu, service_type.value.cpu)

        self.num_tasks = 0
        
    def print_data(self):
        print(f"Pod {self.id} {self.service_type} powerstate: {self.power_state} ram capacity: {self.ram.capacity} ram level: {self.ram.level}")

class Node(object):
    def __init__(self, env: Any, node_id: int, power_state: PowerState) -> None:
        self.env = env
        self.id = node_id
        self.pod_idc = 0

        self.power_state = power_state
        self.ram = simpy.Container(self.env, NODE_RAM, NODE_RAM)
        self.cpu = simpy.Container(self.env, NODE_CORE, NODE_CORE)

        self.num_idle_pods = {st: 0 for st in ServiceType}
        self.num_active_pods = {st: 0 for st in ServiceType}

        self.pods: List[Pod] = []
        self.num_tasks = 0
        
    def average_usage(self, service_type: ServiceType = None):
        avr_usage_sum = 0
        if service_type == None:
            return ((self.ram.capacity - self.ram.level) / self.ram.capacity ) * 100
        for pod in self.pods:
            if pod.service_type == service_type and pod.power_state == PowerState.ON:
                used_ram = pod.ram.capacity - pod.ram.level
                ram_usage = (used_ram / pod.ram.capacity) * 100
                avr_usage_sum += ram_usage
                
        if self.num_active_pods[service_type] == 0:
            avr_usage = 0
        else:
            avr_usage = avr_usage_sum / (self.num_active_pods[service_type])
            
        if(False): #avr_usage == 50
            print(f"usage is 50%!!!!!!!!!!!!!!")
            for pod in self.pods:
                if(pod.power_state == PowerState.ON):
                    pod.print_data()
            
        return avr_usage

    def desired_replicas(self, desired_usage, service_type: ServiceType):
                
        avr_usage = self.average_usage(service_type)
        x = self.num_active_pods[service_type] * (avr_usage / desired_usage)
        
        if x == 0: return 1
        return math.ceil(x)

    def idle_pod_search(self, service_type: ServiceType):       #return priority IDLE > ON > OFF >-1
        idle_pod_id = -1
        on_pod_id = -1
        off_pod_id = -1
        for pod in self.pods:
            if pod.service_type == service_type:
                if pod.power_state == PowerState.IDLE:
                    idle_pod_id = pod.id
                    break
                elif pod.power_state == PowerState.ON:
                    on_pod_id = pod.id
                elif pod.power_state == PowerState.OFF:
                    off_pod_id = pod.id
        if is_id_valid(idle_pod_id):
            return idle_pod_id
        if is_id_valid(on_pod_id):
            return on_pod_id
        return off_pod_id

    def off_pod_search(self, service_type: ServiceType):
        for pod in self.pods:
            if pod.power_state == PowerState.OFF and pod.service_type == service_type:
                return pod.id
        return -1

    def oldest_pod_search(self, service_type: ServiceType) -> Any:
        pod_id = -1
        for pod in self.pods:
            if(pod.service_type == service_type):
                pod_id = pod.id
                break
        for pod in self.pods:
            if pod.birth < self.pods[pod_id].birth and pod.service_type == service_type and pod.power_state == PowerState.ON:
                pod_id = pod.id
        return pod_id


    def least_used_pod(self, service_type: ServiceType, task_ram) -> Any:
        lowest_load_id = -1
        for pod in self.pods:
            if (pod.power_state == PowerState.ON and pod.service_type == service_type): #and pod.ram.level > task_ram
                lowest_load_id = pod.id
                break

        for pod in self.pods:
            lowest_load_pod = self.pods[lowest_load_id]
            if (pod.power_state == PowerState.ON and pod.ram.level < lowest_load_pod.ram.level and pod.service_type == service_type):
                lowest_load_id = pod.id
        return lowest_load_id

    def pod_sync_timeout(self, service_type):
            real_active = sum(1 for pod in self.pods if pod.power_state == PowerState.ON and pod.service_type == service_type)
            while real_active != self.num_active_pods[service_type]:
                yield self.env.timeout(10)
                real_active = sum(1 for pod in self.pods if pod.power_state == PowerState.ON and pod.service_type == service_type)

class Cluster(object):
    def __init__(self, env: Any, number_of_nodes: int,) -> None:
        self.env = env
        self.number_of_nodes = number_of_nodes

        self.nodes = [Node(env, node_id=i, power_state=PowerState.OFF) for i in range(NUMBER_OF_NODES)]
        self.tasks_waiting = 0
        self.prev_tasks_waiting = []

        self.reward = 0
        self.excess_time_digest = TDigest()
        self.excess_time_digest.update(0.)
        

        self.active_nodes = 0
        self.finished_tasks = 0
        self.arrival_rate = ARRIVAL_RATE
        
    def prev_tasks_waiting_mean(self):
        tasks_sum = 0
        for number in self.prev_tasks_waiting:
            tasks_sum += number
        return tasks_sum / max(len(self.prev_tasks_waiting), 1)

    def start_nodes(self, amount: int):
        print(f"node generation started")
        for id in range(amount):
            yield self.env.process(start_off_node(self.env, self, id))

            #yield self.env.process(start_off_pod(self.env, self, id, service_type= ServiceType.typeA))
            #yield self.env.process(start_off_pod(self.env, self, id, service_type= ServiceType.typeB))
            #yield self.env.process(start_off_pod(self.env, self, id, service_type= ServiceType.typeC))
            print(f"node {id} started")
        print(f"all nodes started")


    def search_for_node(self, service_type = None) -> Any:
        lowest_id = -1
        lowest_load_node = self.nodes[0]
        for node in self.nodes:
            if (node.power_state == PowerState.ON):      
                if service_type != None and node.num_active_pods[service_type] > 0:
                    lowest_load_node = node
                    lowest_id = node.id
                    break
                elif service_type == None:
                    lowest_load_node = node
                    lowest_id = node.id
                    break
                
        if is_id_valid(lowest_id):
            for node in self.nodes:
                if (node.power_state == PowerState.ON and lowest_load_node.average_usage(service_type) > node.average_usage(service_type)):
                    if service_type != None and node.num_active_pods[service_type] > 0:
                        lowest_load_node = node
                        lowest_id = node.id
                    elif service_type == None:
                        lowest_load_node = node
                        lowest_id = node.id
        if lowest_id == 1 and service_type != None:
            if lowest_load_node.num_active_pods[service_type] == 0:
                print("tenylegesen returnolve lett 1 ugy hogz nincs aktiv podja")
        return lowest_id
        

    def idle_node_search(self) -> Any:          #return priority: IDLE > OFF > ON
        max_node_id = self.number_of_nodes - 1
        node_id = max_node_id
        for node in self.nodes:
            if node.power_state == PowerState.IDLE:
                node_id = node.id
                break
        if (node_id == max_node_id):
            for node in self.nodes:
                if (node.power_state == PowerState.OFF):
                    node_id = node.id
                    break
        return node_id


    def scale_out(self) -> Any:
        node_id = self.idle_node_search()
        node = self.nodes[node_id]

        if (node_id < self.number_of_nodes):
            if (node.power_state == PowerState.OFF):
                yield self.env.process(start_off_node(self.env, cluster = self, node_id=node_id))
            elif(node.power_state == PowerState.IDLE):
                yield self.env.process(wake_node(self.env, cluster = self, node_id = node_id))       
            #print(f"nodes scaled OUT, active nodes: {self.active_nodes} node id: {node.id}")


    def scale_in(self) -> Any:
        node_id = self.search_for_node()
        #print(f"node returned for scaling in {node_id}")
        if is_id_valid(node_id):
            yield self.env.process(scale_in_node(self.env, cluster = self, node_id = node_id))
            #print(f"Nodes scaled in with node id: {node_id} powerstate: {self.nodes[node_id].power_state} active nodes: {self.active_nodes}")
            
    def calculate_arrival_rate(self, variation: int = 0):
        if(variation == 0): #smooth changes around arrival rate
            self.arrival_rate += random.uniform(-0.2, 0.2)
            self.arrival_rate = min(max(0.2, self.arrival_rate), 1.2)
        elif(variation == 1):    #noise around arrival_rate
            self.arrival_rate = (random.uniform(5,20) + arrival_rate) / 2
            
    def print_data(self, pod_data = False, type_data = False):
        print(f"****************** active nodes: {self.active_nodes} ******************")
        #print(f"this was printed out from task {task_number}, this task is being run on nodes[{task_node_id}].pods[{task_pod_id}]")
        for node in self.nodes:
            if node.power_state == PowerState.ON:
                node_ram_usage = ((node.ram.capacity - node.ram.level) / node.ram.capacity) * 100
                print(f"Node: {node.id} ram usage: {node_ram_usage}% powerstate: {node.power_state}")
                if type_data:
                    print(f"---------- Node: {node.id} with powerstate: {node.power_state}----------")
                    print(f"ram usage for ServiceType A: {node.average_usage(ServiceType.typeA)}%")
                    print(f"ram usage for ServiceType B: {node.average_usage(ServiceType.typeB)}%")
                    print(f"ram usage for ServiceType C: {node.average_usage(ServiceType.typeC)}%")
                print(f"Tasks running at this moment {node.num_tasks}")
                
                if pod_data:
                    print(f"Pods listed below")
                    for pod in node.pods:
                        if(pod.power_state == PowerState.ON):
                            print(f"pod id:{pod.id} available ram:{pod.ram.level} {pod.service_type} powerstate:{pod.power_state}")

def is_id_valid(id: int):
    if(id < 0):
        return False
    else:
        return True

def print_task_data(cluster: Cluster, service_type: ServiceType, task_number, task_node_id, task_pod_id, retry):
    real_active = sum(1 for pod in cluster.nodes[task_node_id].pods if pod.power_state == PowerState.ON and pod.service_type == service_type)
    #if(retry % 2):
        #print(f"task {task_number} is in loop and has retried {retry}")
    if(retry % 1000 == 0 and retry != 0): #retry % 30 == 0
        print('-------------task', task_number,' retried pod search:', retry, 'times, search returned node_id: ', task_node_id, ' pod id:', task_pod_id, ' node powerstate =', cluster.nodes[task_node_id].power_state, 'ram usage: ', cluster.nodes[task_node_id].average_usage(),'%')
        print('number of active pods: ', cluster.nodes[task_node_id].num_active_pods[service_type], 'real active: ', real_active, ' on node:', task_node_id)
        print('compatible pods below')
        for pod in cluster.nodes[task_node_id].pods:
            if service_type == pod.service_type:
                print('nodes pod (id,power,service,task_service): ', pod.id, pod.power_state, pod.service_type, service_type,' node powerstate: ', cluster.nodes[task_node_id].power_state)

def task(env, cluster: Cluster, service_type: ServiceType, task_number):
    task_pod_id = -1
    task_node_id = -1
    retry = 0
    timeout_length = 2
    x = random.uniform(1, 2)
    y = random.uniform(10, 20)
    task_ram = math.ceil(service_type.value.ram/ x)
    task_cpu = service_type.value.cpu / 2
    start = env.now
    while(not is_id_valid(task_pod_id)):
        task_node_id = cluster.search_for_node(service_type)
        if(is_id_valid(task_node_id)):
            task_pod_id = cluster.nodes[task_node_id].least_used_pod(service_type, task_ram)
            
        if not is_id_valid(task_pod_id) and retry != 0:
            yield env.timeout(timeout_length)
            timeout_length = timeout_length * 1.1
        
        print_task_data(cluster, service_type, task_number, task_node_id, task_pod_id, retry)
        
            
        on_nodes_sum = 0
        idle_nodes_sum = 0
        off_nodes_sum = 0
        for node in cluster.nodes:
            if node.power_state == PowerState.ON:
                on_nodes_sum +=1
            elif node.power_state == PowerState.IDLE:
                idle_nodes_sum +=1
            else:
                off_nodes_sum +=1
        
        if on_nodes_sum != cluster.active_nodes:
            print(f"[SYNC ISSUE] Active nodes={on_nodes_sum}, but cluster.active_nodes={cluster.active_nodes} also, Off nodes={off_nodes_sum} Idle nodes={idle_nodes_sum}")
        
        retry += 1
    pod = cluster.nodes[task_node_id].pods[task_pod_id]
    
    cluster.tasks_waiting +=1 
    yield pod.ram.get(task_ram)
    yield pod.cpu.get(task_cpu)
    cluster.tasks_waiting -=1 

    cluster.nodes[task_node_id].num_tasks += 1
    cluster.nodes[task_node_id].pods[task_pod_id].num_tasks += 1

    
    t = random.expovariate(SERVICE_RATE) #???
    #  t=random.uniform(0.01,2)

    
    yield env.timeout(t)
    yield pod.ram.put(task_ram)
    yield pod.cpu.put(task_cpu)
    cluster.nodes[task_node_id].num_tasks -= 1
    cluster.nodes[task_node_id].pods[task_pod_id].num_tasks -= 1
    
    wait_time = retry * timeout_length
    wait_time2 = max(env.now - start - t, 0.1)
    cluster.excess_time_digest.update(wait_time2)
    pod_ram_usage = ((pod.ram.capacity - pod.ram.level) / pod.ram.capacity ) * 100
    
    cluster.reward += 1000
    
    if cluster.tasks_waiting <= cluster.prev_tasks_waiting_mean():
        improvement = "+[IMPROVING]+"      
    else:
        improvement = "-[WORSENING]-"
    
    if task_number % 10000 == 0:
        print(f"Arrival rate: {cluster.arrival_rate:.3f}, {improvement} Tasks waiting: {cluster.tasks_waiting}, previous 25 mean: {cluster.prev_tasks_waiting_mean()}. Active nodes: {cluster.active_nodes} It's excess time is: {cluster.excess_time_digest.percentile(80):.3f}---------")
    
    if pod_ram_usage > 100:
        print(f"[WARNING]: POD {pod.id} on node {task_node.id} exceeded 100%")
    if(task_number == -1):
        print(f"task {task_number} has finished in time of {env.now - start}, pod ram usage: {pod_ram_usage}% node id: {task_node_id} node ram usage: {cluster.nodes[task_node_id].average_usage()}%, active nodes: {cluster.active_nodes}")
    cluster.finished_tasks +=1
    if(task_number == -1):
        print(f"task {task_number} has finished in time of {env.now - start}, node id: {task_node_id} node ram usage: {cluster.nodes[task_node_id].average_usage(), }%")
        print(f"pod id {task_pod_id} pod ram usage: {pod_ram_usage}% ")

def wake_pod(cluster: Cluster, node_id, pod_id):
    pod = cluster.nodes[node_id].pods[pod_id]
    
    if(pod.power_state == PowerState.IDLE):
        pod.power_state = PowerState.ON
        cluster.nodes[node_id].num_active_pods[pod.service_type] += 1
        cluster.nodes[node_id].num_idle_pods[pod.service_type] -= 1
        
    else:
        print('wake pod was called with non idle powerstate, powerstate:', pod.power_state )

def wake_node(env, cluster: Cluster, node_id):
    node = cluster.nodes[node_id]
    if(node.power_state == PowerState.IDLE):
        yield env.timeout(WAKE_TIME_NODE)
        node.power_state = PowerState.ON
        cluster.active_nodes += 1

def start_off_node(env, cluster: Cluster, node_id):
    yield env.timeout(BOOT_TIME_NODE)
    cluster.nodes[node_id].power_state = PowerState.ON
    cluster.active_nodes += 1
    yield env.process(start_off_pod(env, cluster, node_id, ServiceType.typeA))
    yield env.process(start_off_pod(env, cluster, node_id, ServiceType.typeB))
    yield env.process(start_off_pod(env, cluster, node_id, ServiceType.typeC))

def start_off_pod(env, cluster: Cluster, node_id: int, service_type: ServiceType, pod_id = -1):
    node = cluster.nodes[node_id]
    pod_started = False
    if(is_id_valid(pod_id) and node.pods[pod_id].power_state == PowerState.OFF):
        yield env.timeout(BOOT_TIME_POD)
        node.pods[pod_id].power_state = PowerState.ON
        yield node.ram.get(service_type.value.ram)
        yield node.cpu.get(service_type.value.cpu)
        node.num_active_pods[service_type] += 1
    else:
        for pod in node.pods:
            if(pod.service_type == service_type and pod.power_state == PowerState.OFF):
                yield env.timeout(BOOT_TIME_POD)
                pod.power_state = PowerState.ON
                yield node.ram.get(service_type.value.ram)
                yield node.cpu.get(service_type.value.cpu)
                pod_started = True
                node.num_active_pods[service_type] += 1
                break
        if not pod_started:
            yield env.process(new_pod(env, cluster, node_id, service_type, PowerState.ON))

def make_pod_idle(cluster: Cluster, node_id: int, pod_id:int):
    node = cluster.nodes[node_id]
    pod = node.pods[pod_id]
    power_state = pod.power_state
    if power_state == PowerState.ON:
        node.num_active_pods[pod.service_type] -= 1
            
    if power_state != PowerState.IDLE:
        if power_state == PowerState.OFF:
            service_type = node.pods[pod_id].service_type
            yield node.ram.get(service_type.value.ram)
            yield node.cpu.get(service_type.value.cpu)
            
        pod.power_state = PowerState.IDLE
        node.num_idle_pods[pod.service_type] += 1
            



def make_node_idle(env, cluster: Cluster, node_id: int):
    node = cluster.nodes[node_id]
    if node.power_state != PowerState.IDLE:
        if node.power_state == PowerState.ON:
            cluster.active_nodes -= 1
        node.power_state = PowerState.IDLE
        for pod in node.pods:
            yield env.process(make_pod_idle(cluster = cluster, node_id = node_id, pod_id = pod.id))


def new_pod(env, cluster: Cluster, node_id: int, service_type: ServiceType, power_state: PowerState):   
    pod = Pod(env, cluster.nodes[node_id].pod_idc, service_type, power_state)
    cluster.nodes[node_id].pod_idc += 1
    cluster.nodes[node_id].pods.append(pod)

    if(power_state != PowerState.OFF ):
        yield env.timeout(BOOT_TIME_POD)
        yield cluster.nodes[node_id].ram.get(service_type.value.ram)
        yield cluster.nodes[node_id].cpu.get(service_type.value.cpu)
        
        if power_state == PowerState.ON:
            cluster.nodes[node_id].num_active_pods[service_type] +=1
        elif power_state == PowerState.IDLE:
            cluster.nodes[node_id].num_idle_pods[service_type] +=1

def terminate_pod(env, cluster: Cluster, node_id, pod_id):
        node = cluster.nodes[node_id]
        pod = node.pods[pod_id]
        service_type = pod.service_type
        
        if pod.power_state != PowerState.OFF:
            if pod.power_state == PowerState.ON:
                node.num_active_pods[service_type] -=1
            if pod.power_state == PowerState.IDLE:
                node.num_idle_pods[service_type] -=1
            pod.power_state = PowerState.OFF
            #print('pod', pod.id, ' was just terminated')
            while pod.num_tasks != 0:
                yield env.timeout(1)
                
            

            yield node.ram.put(service_type.value.ram)
            yield node.cpu.put(service_type.value.cpu)


def scale_in_node(env, cluster: Cluster, node_id):
    yield env.process(make_node_idle(env, cluster = cluster, node_id = node_id))
    for pod in cluster.nodes[node_id].pods:
        env.process(terminate_pod(env, cluster, node_id, pod.id))



def task_generator(env, cluster):
    print('Task generation has started') 
    env.timeout(CLUSTER_CONTROL_TIME)
    for i in itertools.count():
        service_type_map = [
            ServiceType.typeA,
            ServiceType.typeB,
            ServiceType.typeC,
        ]
        task_type = i % 3
        service_type = service_type_map[task_type]
        
        if i % 10000 == 0:
            cluster.calculate_arrival_rate(VARIATION)
        #cluster.arrival_rate = 0.6
        t = random.expovariate(cluster.arrival_rate)
      
        #cluster.arr_digest.update(t)
        yield env.timeout(t)
        env.process(task(env, cluster, service_type, i))

        #if (i % 30 ==0) :
           #print('task generated with number:', i, ", time: ", env.now, ', service type is: ', service_type,)

def monitor(env, task_gen, cluster_con, a, b, c):
    while True:
        if not task_gen.is_alive:
            print(f"[FATAL ERROR]: Task generator ended")
        if not cluster_con.is_alive:
            print(f"[FATAL ERROR]: Cluster control ended")
        if not a.is_alive:
            print(f"[FATAL ERROR]: Pod scale typeA ended")
        if not b.is_alive:
            print(f"[FATAL ERROR]: Pod scale typeB ended")
        if not c.is_alive:
            print(f"[FATAL ERROR]: Pod scale typeC ended")

        yield env.timeout(100)   # check every 1 time unit

class ClusterEnv(ExternalEnv):
    def __init__(self, config: EnvContext):
        self.number_of_nodes = config["number_of_nodes"]
        self.percentile_points = config["percentile_points"]
        self.scale_running = {stype: False for stype in ServiceType}
        obs_space1 = Tuple(
          [
            Discrete(self.number_of_nodes + 1),#   active nodes
            
            Box(0, np.inf, shape=(1,),dtype=np.float64), #arrival rate
 
            #Box(0, NODE_RAM, shape=(self.number_of_nodes,),dtype=np.float64),#   ram usage
            
            Box(0, np.inf, shape=(1,),dtype=np.float64), #task wait time percentile
            
            Box(0, np.inf, shape=(1,),dtype=np.float64), #tasks waiting
            
            Box(0, np.inf, shape=(1,),dtype=np.float64), #previous waiting tasks mean
          ]
        )
        obs_space2 = Discrete(self.number_of_nodes + 1)
        self.observation_space = obs_space1
        self.action_space = Discrete(3)
        env_config = {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }
        ExternalEnv.__init__(self, self.action_space, self.observation_space)
        self.k8env = simpy.Environment()
        self.cluster = Cluster(self.k8env, self.number_of_nodes)
        self.loop = 0
        
        self.ext = np.zeros(shape=(self.percentile_points,))
        #self.res = np.zeros(shape=(self.percentile_points,))
        #self.tas = np.zeros(shape=(self.percentile_points,))
        
        self.nodes_ram_usage = np.zeros(shape=(self.number_of_nodes,))
        self.nodes_task_number = np.zeros(shape=(self.number_of_nodes,))
        

    def scale_in_pods(self, node: Node, service_type: ServiceType):
        chosen_pod_id = node.oldest_pod_search(service_type)
        if(is_id_valid(chosen_pod_id)):
            yield self.k8env.process(make_pod_idle(self.cluster, node_id = node.id, pod_id = chosen_pod_id,))
        yield self.k8env.timeout(5)

    def scale_out_pods(self, node: Node, service_type: ServiceType):
        idle_pod_id = node.idle_pod_search(service_type)
        if is_id_valid(idle_pod_id):
            pod = node.pods[idle_pod_id]
            
            if(pod.power_state == PowerState.IDLE):
                wake_pod(self.cluster, node.id, idle_pod_id)
            if(pod.power_state == PowerState.OFF):
                yield self.k8env.process(start_off_pod(self.k8env, self.cluster, node.id, service_type, idle_pod_id))            

        elif idle_pod_id == -1:
            yield self.k8env.process(start_off_pod(self.k8env, self.cluster, node.id, service_type, idle_pod_id))
        yield self.k8env.timeout(5)  
        
    def pod_idle_control(self, node: Node, service_type: ServiceType):
        if node.num_idle_pods[service_type] < math.ceil(node.num_active_pods[service_type] * IDLE_ACTIVE_RATIO):
            off_pod_id = node.off_pod_search(service_type)
            if is_id_valid(off_pod_id):
                yield self.k8env.process(make_pod_idle(self.cluster, node.id, off_pod_id))
            else:
                yield self.k8env.process(new_pod(self.k8env, cluster = self.cluster, node_id = node.id, service_type = service_type, power_state = PowerState.IDLE))

        elif node.num_idle_pods[service_type] > math.ceil(node.num_active_pods[service_type] * IDLE_ACTIVE_RATIO):
            #lehet yield kene terminationhoz
            idle_pod_id = node.idle_pod_search(service_type)
            if is_id_valid(idle_pod_id) and node.pods[idle_pod_id].power_state == PowerState.IDLE:
                self.k8env.process(terminate_pod(env = self.k8env, cluster = self.cluster, node_id = node.id, pod_id = idle_pod_id))

    def scale_pods_by_usage(self, service_type: ServiceType):
        yield self.k8env.timeout(CLUSTER_CONTROL_TIME*10)
        try:
            
            while True:
                for node in self.cluster.nodes:
                    if node.power_state == PowerState.ON:
                        #if node.id == 1:
                            #print(f"node {node.id} scaling pods rn, active pods with type {service_type}:{node.num_active_pods[service_type]}, idle pods:{node.num_idle_pods[service_type]}|||||||||||")
                        yield self.k8env.process(node.pod_sync_timeout(service_type))
                        desired_pods = node.desired_replicas(POD_USAGE, service_type)
                        
                        if node.num_active_pods[service_type] > desired_pods:
                            yield self.k8env.process(self.scale_in_pods(node, service_type))
                        elif node.num_active_pods[service_type] < desired_pods:
                            #if node.id == 1:
                                #print(f"should scale out pods for node{node.id}|||||||||||")
                            yield self.k8env.process(self.scale_out_pods(node, service_type))

                    yield self.k8env.process(self.pod_idle_control(node, service_type))
                yield self.k8env.timeout(10)
        finally:
            self.scale_running[service_type] = False


    def step(self, action):
        print("STEP?")
        return 100, 100, 100, {}

    def digestor(self):
        self.cluster.excess_time_digest.update(0.)
        for i in range(self.percentile_points):
            if self.cluster.excess_time_digest.n == 0:
                #self.arr[i] = 0
                self.ext[i] = 0
                print('ext got 0 datapoints')
            else:
                self.ext[i] = max(0.0, self.cluster.excess_time_digest.percentile(i))
                

    def reward_calculator(self):
        excess_time = max(0.001, self.cluster.excess_time_digest.percentile(80.))
        wait_time_bonus = 150000/ ((excess_time + 1)**0.5)
        improvement_bonus = max((self.cluster.prev_tasks_waiting_mean() - self.cluster.tasks_waiting) * 500, 20000)
        
        reward = wait_time_bonus + self.cluster.reward
        return reward
        
    def punishment_calculator(self):
        punishment = 0
        multiplier_node_activity = 2 - 1 * (self.cluster.active_nodes/ NUMBER_OF_NODES) #(NUMBER_OF_NODES / self.cluster.active_nodes)
        multiplier_task_wait = 2- math.exp(- 0.00001 * self.cluster.tasks_waiting)
        excess_time = max(0.001, self.cluster.excess_time_digest.percentile(80.))
        #multiplier = 10/self.cluster.active_nodes 
        
        if self.cluster.tasks_waiting <= self.cluster.prev_tasks_waiting_mean():
            return 0
        
        if self.action != Action.ScaleOut:
            punishment = excess_time / 700
        else:
            punishment = excess_time / 3500
        punishment = punishment * 300 * multiplier_node_activity * multiplier_task_wait
        #print(f"Punishment: {punishment:.3f}. Node activity multiplier: { multiplier_node_activity:.3f} task wait multiplier: {multiplier_task_wait:.3f} tasks waiting:{self.cluster.tasks_waiting:.3f}")
        return punishment
        
    def observation_calculator(self):
        cluster = self.cluster
        wait_time_percentile = cluster.excess_time_digest.percentile(80.)
        waiting_tasks_val = cluster.tasks_waiting
        waiting_tasks_mean_val = cluster.prev_tasks_waiting_mean()
        
        arrival_array = np.array([cluster.arrival_rate], dtype=np.float64)
        wait_time_array = np.array([wait_time_percentile], dtype=np.float64)
        waiting_tasks_array = np.array([waiting_tasks_val], dtype=np.float64)
        waiting_tasks_mean_array = np.array([waiting_tasks_val], dtype=np.float64)
        
        obs = tuple([ self.cluster.active_nodes, arrival_array, wait_time_array, waiting_tasks_array, waiting_tasks_mean_array]) #self.res, self.arr, self.tas, , self.nodes_ram_usage
        obs2 = self.cluster.active_nodes
        if (isinstance(self.observation_space, Discrete)):
            return obs2
        else:
            return obs

    def cluster_control(self):
        self.time_start = time.time()
        self.cluster.response_digest = TDigest()
        #self.cluster.arr_digest = TDigest()
        print('Cluster control has started, TDigest reset')
        yield self.k8env.process(self.cluster.start_nodes(4))
        #print(self.cluster.active_nodes)
        print(f"before cluster control timeout")
        yield self.k8env.timeout(CLUSTER_CONTROL_TIME)
        print(f"after cluster control timeout")
        print('digestor 1st run')
        self.digestor()
        
        
        
        #try:
        while True:
            self.cluster.reward = 0
            self.loop +=1
            self.episode_id = self.start_episode()
            self.action = self.action_space.sample()
            
            if len(self.cluster.prev_tasks_waiting) > 25:
                self.cluster.prev_tasks_waiting.pop(0)
            
            for node in self.cluster.nodes:
                self.nodes_ram_usage[node.id] = node.ram.capacity - node.ram.level
                self.nodes_task_number[node.id] = node.num_tasks
            
            self.digestor()
            
            if self.action == Action.ScaleOut and self.cluster.active_nodes < self.cluster.number_of_nodes:
                yield self.k8env.process(self.cluster.scale_out())
            elif self.action == Action.ScaleIn and self.cluster.active_nodes > 1:
                yield self.k8env.process(self.cluster.scale_in())
 
            del self.cluster.excess_time_digest
            self.cluster.excess_time_digest = TDigest()
            yield self.k8env.timeout(CLUSTER_CONTROL_TIME)

            for node in self.cluster.nodes:
                self.nodes_ram_usage[node.id] = node.ram.capacity - node.ram.level
                self.nodes_task_number[node.id] = node.num_tasks

            if self.cluster.active_nodes == 0:
                reward = -10
            else:
                reward = self.reward_calculator()
                punishment = self.punishment_calculator()
                
            self.cluster.prev_tasks_waiting.append(self.cluster.tasks_waiting)
            
                
            if(self.loop % 500 == 0) :
                excess_time = self.cluster.excess_time_digest.percentile(80.)
                time_elapsed = time.time() - self.time_start
                #print(f"------digestor loop: {self.loop} number of finished tasks: {self.cluster.finished_tasks} time: {self.k8env.now} real time elapsed: {time_elapsed} seconds")
                print(f"t{self.cluster.finished_tasks}--active_nodes: {self.cluster.active_nodes} excess time was: {excess_time} ----- loop reward: {(reward - punishment):.3f} ----- r: {reward:.3f} p: {-punishment:.3f}\n")
                #arr = [percentile for percentile in self.cluster.excess_time_digest]
                #print(*arr)
                
                #self.cluster.print_data()
                
            info = ""
            observation = self.observation_calculator()
            self.log_action(self.episode_id, observation, self.action)
            self.log_returns(self.episode_id, reward - punishment, info=info)
            self.end_episode(self.episode_id, observation)
            if(self.loop % 1000 == 0) :
                print(f"Iteration over, time passed: { (time.time() - self.time_start):.3f} seconds")
            
        #except Exception as e:
            #print("cluster control caught", repr(e))
            

    def run(self):
        cluster_con = self.k8env.process(self.cluster_control())
        task_gen = self.k8env.process(task_generator(self.k8env, self.cluster))

        a = self.k8env.process(self.scale_pods_by_usage(ServiceType.typeA))
        b = self.k8env.process(self.scale_pods_by_usage(ServiceType.typeB))
        c = self.k8env.process(self.scale_pods_by_usage(ServiceType.typeC))
        self.k8env.process(monitor(self.k8env, task_gen, cluster_con, a, b, c))

        print("--Starting simulation--")
        self.k8env.run(until=SIM_TIME)
        print("--Simulation ended--")

def env_creator(env_config):
     return ClusterEnv(env_config)



if __name__ == "__main__":

    random.seed(RANDOM_SEED)
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode,num_gpus=1)

    register_env("k8-env", lambda _: ClusterEnv(config={
     "number_of_nodes": NUMBER_OF_NODES, "percentile_points": 100}))


    config = (
            DQNConfig()
            .environment("k8-env")
            .rollouts(num_rollout_workers=6, enable_connectors=False)
    )
    dqn = config.build()



    #  for _ in framework_iterator(config, frameworks=("torch")):
    #  dqn = config.build()
    for i in range(300):
                result = dqn.train()
                print(
                    f"Iteration {i}, mean reward {result['episode_reward_mean']:.3f}, timesteps {result['timesteps_total']}"
                )
                if i % 20 == 0:
                    checkpoint = dqn.save()
                    #print("Checkpoint saved at", checkpoint)
    
    ray.shutdown()
