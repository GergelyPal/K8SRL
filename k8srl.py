"""
Groups of servers example

Covers:

- Resources: Resource
- Resources: Container
- Waiting for other processes

Scenario:
  A cluster has a number of hosts. Each host has a number of servers.
  Customers randomly arrive at the cluster, request one
  server  and obtain service from that server.

  A cluster control process manages the resource (switch on and off the host).

"""
import argparse
import os
import ray
import math
from ray.rllib.env.env_context import EnvContext
#  from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.algorithms import dqn
from ray.rllib.algorithms.dqn import DQNConfig
#  from ray.rllib.agents.dqn.dqn.DQNTrainer import DQNTrainer
#  from ray.rllib.agents.pg import PGTrainer
#  from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
#  from ray.rllib.utils.test_utils import framework_iterator
from ray.tune.registry import get_trainable_cls
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from typing import Any
#  import gym
from gymnasium.spaces import Tuple, Box, Discrete
import numpy as np
from typing import List
import sys

import simpy
import random

from tdigest import TDigest
from enum import IntEnum, Enum
from collections import namedtuple
import itertools

RANDOM_SEED = 42

BOOT_TIME_NODE = 20
WAKE_TIME_NODE = 1
BOOT_TIME_POD = 1

IDLE_ACTIVE_RATIO = 0.2

SIM_TIME = 110000000            # Simulation time in seconds
NUM_ARRIVALS = 10000
CLUSTER_CONTROL_TIME = 5

NUMBER_OF_NODES = 10
NODE_RAM = 16000
NODE_CORE = 8
POD_USAGE = 90

ARRIVAL_RATE = 20.
SERVICE_RATE = 2.0
SERVICE_TIME = 1.2
RESPONSE_TIME_THRESHOLD_D = 1.3
RESPONSE_TIME_THRESHOLD_U = 4.0



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
pod_idc = 0

class ServiceType(Enum):
    typeA = Resources(ram = 128, cpu = 0.1)
    typeB = Resources(ram = 256, cpu = 0.2)
    typeC = Resources(ram = 512, cpu = 0.4)

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

class Node(object):
    def __init__(self, env: Any, node_id: int, power_state: PowerState) -> None:
        self.env = env
        self.id = node_id

        self.power_state = power_state
        self.ram = simpy.Container(self.env, NODE_RAM, NODE_RAM)
        self.cpu = simpy.Container(self.env, NODE_CORE, NODE_CORE)

        self.num_idle_pods = 0
        self.num_active_pods = 0

        self.pods: List[Pod] = []
        self.num_tasks = 0

    def desired_replicas(self, desired_usage, service_type: ServiceType):
        avr_usage = 0
        for pod in self.pods:
            if pod.service_type == service_type:
                used_ram = pod.ram.capacity - pod.ram.level
                ram_usage = used_ram / pod.ram.capacity * 100
                avr_usage += ram_usage

        avr_usage = avr_usage / len(self.pods)
        x = len(self.pods) * (avr_usage / desired_usage)
        return math.ceil(x)

    def idle_pod_search(self) -> Any:       #return priority IDLE > ON > OFF
        pod_id = len(self.pods)
        for pod in self.pods:
            if pod.power_state == PowerState.IDLE:
                return pod.id

        for pod in self.pods:
            if (pod.power_state == PowerState.ON):
                pod_id = pod.id
                break
        return pod_id

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


    def least_used_pod(self, service_type: ServiceType) -> Any:
        lowest_load_pod_id = 0
        pod_found = False
        for pod in self.pods:
            if (pod.power_state == PowerState.ON and pod.service_type == service_type):
                lowest_load_pod_id = pod.id
                pod_found = True
                break
        if not pod_found:
            return -1

        for pod in self.pods:
            lowest_load_pod = self.pods[lowest_load_pod_id]
            if (pod.power_state == PowerState.ON and pod.ram.level < lowest_load_pod.ram.level and pod.service_type == service_type):
                lowest_load_node_id = pod.id
        return lowest_load_node_id

class Cluster(object):
    def __init__(self, env: Any, number_of_nodes: int,) -> None:
        self.env = env
        self.number_of_nodes = number_of_nodes

        self.nodes = [Node(env, node_id=i, power_state=PowerState.OFF) for i in range(NUMBER_OF_NODES)]

        for node in self.nodes:
            if node.power_state == PowerState.ON:
                node.pods.append(Pod(env, len(self.nodes), ServiceType.TypeA, PowerState.ON))
                node.pods.append(Pod(env, len(self.nodes), ServiceType.TypeB, PowerState.ON))
                node.pods.append(Pod(env, len(self.nodes), ServiceType.TypeC, PowerState.ON))

        self.digest = TDigest()     #  response time
        self.arrdigest = TDigest()  #  arrival time
        self.arrdigest.update(100.)
        self.digest.update(0.)

        self.active_nodes = 0

    def start_nodes(self, amount: int):
        for id in range(amount):
            yield self.env.process(start_off_node(self.env, self, id))

    def search_for_node(self) -> Any:
        lowest_load_node_id = 0
        for node_id in range(self.number_of_nodes):
            if (self.nodes[node_id].power_state == PowerState.ON):
                lowest_load_node_id = node_id
                break
        for node_id in range(self.number_of_nodes):
            node = self.nodes[node_id]
            lowest_load_node = self.nodes[lowest_load_node_id]
            if (node.power_state == PowerState.ON and node.ram > lowest_load_node.ram):
                lowest_load_node_id = node_id
        return lowest_load_node_id

    def idle_node_search(self) -> Any:          #return priority: IDLE > OFF > ON
        max_node_id = self.number_of_nodes - 1
        node_id = max_node_id
        for j in range(self.number_of_nodes):
            if self.nodes[j].power_state == PowerState.IDLE:
                node_id = j
                break
        if (node_id == max_node_id):
            for i in range(self.number_of_nodes):
                if (self.nodes[j].power_state == PowerState.OFF):
                    node_id = i
                    break
        return node_id


    def scale_out(self) -> Any:
        node_id = self.idle_node_search()
        node = self.nodes[node_id]

        if (node_id < self.number_of_nodes):
            if (node.power_state == PowerState.OFF):
                yield self.env.process(start_off_node(self.env, self.cluster, node_id))
            elif(node.power_state == PowerState.IDLE):
                yield self.env.process(wake_node(self.env, cluster = self, node_id = node_id))


    def scale_in(self, env) -> Any:
        node_id = self.least_pods_node_search()
        yield env.process(scale_in_node(env, cluster = self, node_id = node_id))

def is_id_valid(id: int):
    if(id < 0):
        return False
    else:
        return True

#TODO mindig legyen egy az idle nodeok es podok kozul kikapcoslva a megadott arany
def task(env, cluster: Cluster, service_type: ServiceType):
    task_pod_id = -1
    while(not is_id_valid(task_pod_id)):
        task_node_id = cluster.search_for_node()
        task_pod_id = cluster.nodes[task_node_id].least_used_pod(service_type)

    yield cluster.nodes[task_node_id].pods[task_pod_id].ram.get(service_type.value.ram)
    yield cluster.nodes[task_node_id].pods[task_pod_id].cpu.get(service_type.value.cpu)
    cluster.nodes[task_node_id].num_tasks += 1
    cluster.nodes[task_node_id].pods[task_pod_id].num_tasks += 1

    start = env.now
    t = random.expovariate(SERVICE_RATE) #???
    #  t=random.uniform(0.01,2)

    
    yield env.timeout(t)
    yield cluster.nodes[task_node_id].pods[task_pod_id].ram.put(service_type.value.ram)
    yield cluster.nodes[task_node_id].pods[task_pod_id].cpu.put(service_type.value.cpu)
    cluster.nodes[task_node_id].num_tasks -= 1
    cluster.nodes[task_node_id].pods[task_pod_id].num_tasks -= 1

    cluster.digest.update(env.now - start)

def wake_pod(cluster: Cluster, node_id, pod_id):
    pod = cluster.nodes[node_id].pods[pod_id]
    if(pod.power_state == PowerState.IDLE):
        pod.power_state = PowerState.ON
        cluster.nodes[node_id].num_active_pods += 1
        cluster.nodes[node_id].num_idle_pods -= 1

def wake_node(env, cluster: Cluster, node_id):
    node = cluster.nodes[node_id]
    if(node.power_state == PowerState.IDLE):
        yield env.timeout(WAKE_TIME_NODE)
        node.power_state = PowerState.ON

def start_off_node(env, cluster: Cluster, node_id):
    yield env.timeout(BOOT_TIME_NODE)
    cluster.nodes[node_id] = PowerState.ON
    cluster.active_node += 1
    yield env.process(start_off_pod(env, cluster, node_id, ServiceType.typeA))
    yield env.process(start_off_pod(env, cluster, node_id, ServiceType.typeB))
    yield env.process(start_off_pod(env, cluster, node_id, ServiceType.typeC))

def start_off_pod(env, cluster: Cluster, node_id: int, service_type: ServiceType, pod_id = -1):
    node = cluster.nodes[node_id]
    if(is_id_valid(pod_id) and node.pods[pod_id].power_state == PowerState.OFF):
        yield env.timeout(BOOT_TIME_POD)
        node.pods[pod_id].power_state = PowerState.ON
        yield node.ram.get(service_type.value.ram)
        yield node.cpu.get(service_type.value.cpu)
    else:
        for pod in node.pods:
            if(pod.service_type == service_type and pod.power_state == PowerState.OFF):
                yield env.timeout(BOOT_TIME_POD)
                pod.power_state = PowerState.ON
                yield node.ram.get(service_type.value.ram)
                yield node.cpu.get(service_type.value.cpu)
                node.num_active_pods += 1

                return
        yield env.process(new_pod(env, cluster, node_id, service_type, PowerState.ON))
    node.num_active_pods += 1

def make_pod_idle(cluster: Cluster, node_id: int, pod_id:int):
    node = cluster.nodes[node_id]
    if(node.pods[pod_id].power_state == PowerState.ON):
        node.num_active_pods -= 1
    if(node.pods[pod_id].power_state != PowerState.IDLE):
        node.pods[pod_id].power_state = PowerState.IDLE
        node.num_idle_pods += 1

def make_node_idle(cluster: Cluster, node_id: int):
    node = cluster.nodes[node_id]
    if(node.power_state != PowerState.IDLE):
        node.power_state = PowerState.IDLE


def new_pod(env, cluster: Cluster, node_id: int, service_type: ServiceType, power_state: PowerState):
    global pod_idc
    yield env.timeout(BOOT_TIME_POD)
    pod = Pod(env, pod_idc, service_type, power_state)
    pod_idc += 1
    cluster.nodes[node_id].pods.append(pod)

    if(power_state != PowerState.OFF ):
        yield cluster.nodes[node_id].ram.get(service_type.value.ram)
        yield cluster.nodes[node_id].cpu.get(service_type.value.cpu)

def terminate_pod(env, cluster: Cluster, node_id, pod_id):
    node = cluster.nodes[node_id]
    pod = node.pods[pod_id]
    service_type = pod.service_type

    pod.power_state = PowerState.OFF
    while pod.num_tasks != 0:
        yield env.timeout(1)

    yield node.ram.put(service_type.value.ram)
    yield node.cpu.put(service_type.value.cpu)

    
    


def scale_in_node(env, cluster: Cluster, node_id):
    make_node_idle(cluster, node_id)
    for pod in cluster.nodes[node_id].pods:
        env.process(terminate_pod(env, cluster, node_id, pod.id))

    cluster.active_node -= 1

def task_generator(env, cluster):
    for i in itertools.count():
        service_type_map = [
            ServiceType.TypeA,
            ServiceType.TypeB,
            ServiceType.TypeC,
        ]
        task_type = i % 3
        service_type = service_type_map[task_type]
        if env.now < SIM_TIME/2:
            t = random.expovariate(ARRIVAL_RATE)
        elif env.now < SIM_TIME*3/4:
            t = random.expovariate(ARRIVAL_RATE/2)
        else:
            t = random.expovariate(ARRIVAL_RATE/4)
        cluster.arrdigest.update(t)
        yield env.timeout(t)
        env.process(task(env, cluster, service_type))

        if (i % 20000 ==0) :
           print('task', i, "time --", env.now, 'service type is:', service_type,)

class ClusterEnv(ExternalEnv):
    def __init__(self, config: EnvContext):
        self.number_of_nodes = config["number_of_nodes"]
        self.percentile_points = config["percentile_points"]
        self.scale_running = False
        self.observation_space = Tuple(
          [
            Box(0, np.inf, shape=(self.percentile_points,),dtype=np.float64),# Arrival cdf

            Box(0, np.inf, shape=(self.percentile_points,),dtype=np.float64),# response

            Box(0, NODE_RAM, shape=(self.number_of_nodes,),dtype=np.int64),#   ram usage

            Box(0, np.inf, shape=(self.number_of_nodes,),dtype=np.int64),#     number of tasks on each node
          ]
        )
        print("Initialization ---")
        self.action_space = Discrete(3)
        env_config = {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }
        ExternalEnv.__init__(self, self.action_space, self.observation_space)
        self.k8env = simpy.Environment()
        self.cluster = Cluster(self.k8env, self.number_of_nodes)
        self.loop = 0
        self.arr = np.zeros(shape=(self.percentile_points,))
        self.ser = np.zeros(shape=(self.percentile_points,))

    def scale_down_pods(self, node: Node, service_type: ServiceType):
        chosen_pod_id = node.oldest_pod_search(service_type)
        if(is_id_valid(chosen_pod_id)):
            self.k8env.process(terminate_pod(env = self.k8env, cluster = self.cluster, node_id = node.id, pod_id = chosen_pod_id))

        if(node.num_idle_pods > math.ceil(node.num_active_pods * IDLE_ACTIVE_RATIO)):
            pod_id = node.oldest_pod_search(service_type)
            if(is_id_valid(pod_id)):
                make_pod_idle(self.cluster, node_id = node.id, pod_id = pod_id,)

    def scale_up_pods(self, node: Node, service_type: ServiceType):
        idle_pod_id = node.idle_pod_search()
        pod = node.pods[idle_pod_id]

        if(pod.power_state == PowerState.IDLE):
            wake_pod(self.cluster, node.id, idle_pod_id)

        else:
            yield self.k8env.process(start_off_pod(self.k8env, node.id, service_type))

        if(node.num_idle_pods < math.ceil(node.num_active_pods * IDLE_ACTIVE_RATIO)):
            pod = node.idle_pod_search()
            wake_pod(self.cluster, node.id, pod.id)

    def scale_pods_by_usage(self, service_type: ServiceType):
        try:
            while True:
                for node in self.cluster.nodes:
                    desired_pods = node.desired_replicas(POD_USAGE, service_type)
                    while(desired_pods != len(node.pods)):
                        if len(node.pods) > desired_pods:
                            self.k8env.process(self.scale_down_pods(node, service_type))

                        elif len(node.pods) < desired_pods:
                            yield self.k8env.process(self.scale_up_pods(node, service_type))

                #yield self.k8env.timeout(1)
        finally:
            self.scale_running = False


    def step(self, action):
        print("STEP?")
        return 100, 100, 100, {}

    def cluster_control(self):
        #yield self.k8env.timeout(CONTROL_TIME_INTERVAL)
        for i in range(self.percentile_points):
            self.arr[i] = self.cluster.arrdigest.percentile(i)
            self.ser[i] = self.cluster.digest.percentile(i)

        nodes_ram_usage = []
        nodes_task_num = []
        for node in self.cluster.nodes:
            nodes_ram_usage.append(node.ram.level)
            nodes_task_num.append(node.num_tasks)
        obs = tuple([self.arr, self.ser, nodes_ram_usage, nodes_task_num])

        while True:
            if(self.loop % 100 == 0) :
               print("time", self.k8env.now)
            self.loop +=1
            self.episode_id = self.start_episode()
            self.action = self.action_space.sample()

            if(self.action == Action.ScaleOut and self.cluster.active_node == self.cluster.number_of_nodes): #or  (self.action == Action.ScaleIn and self.cluster.active_node == 1)
                self.action = Action.Do_nothing

            elif self.action == Action.ScaleOut:
                yield self.k8env.process(self.cluster.scale_out)
            elif self.action == Action.ScaleIn:
                yield self.k8env.process(self.cluster.scale_in)

            self.log_action(self.episode_id, obs, self.action)

            #  reward here and observation
            del self.cluster.digest
            self.cluster.digest = TDigest()
            del self.cluster.arrdigest
            self.cluster.arrdigest = TDigest()

            yield self.k8env.timeout(CLUSTER_CONTROL_TIME)

            for i in range(self.percentile_points):
                self.arr[i] = self.cluster.arrdigest.percentile(i)
                self.ser[i] = self.cluster.digest.percentile(i)
            for node in self.cluster.nodes:
                node_id = node.id
                nodes_ram_usage[node_id](node.ram)
                nodes_task_num[node_id](node.num_tasks)

            obs = tuple([self.arr, self.ser, nodes_ram_usage, nodes_task_num])
            excess_time = RESPONSE_TIME_THRESHOLD_U - self.cluster.digest.percentile(99.9)
            if (self.cluster.digest.percentile(99.9) >
                    RESPONSE_TIME_THRESHOLD_U):
                reward = -1 * math.log2(abs(excess_time))/2 * (self.action + 1)
            else:
                reward = excess_time * (self.action + 1) + 2
            info = ""
            self.log_returns(self.episode_id, reward, info=info)
            self.end_episode(self.episode_id, obs)
            del self.cluster.digest
            self.cluster.digest = TDigest()
            del self.cluster.arrdigest
            self.cluster.arrdigest = TDigest()

    def start_pod_scaler(self, service_type: ServiceType):
        while True:
            if not self.scale_running:
                self.scale_running = True
                yield self.k8env.process(self.scale_pods_by_usage(service_type))

    def run(self):
        self.k8env.process(self.cluster_control())
        self.k8env.process(task_generator(self.k8env, self.cluster))

        self.k8env.process(self.start_pod_scaler(ServiceType.typeA))
        self.k8env.process(self.start_pod_scaler(ServiceType.typeB))
        self.k8env.process(self.start_pod_scaler(ServiceType.typeC))

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
            .rollouts(num_rollout_workers=1, enable_connectors=False)
    )
    dqn = config.build()



    #  for _ in framework_iterator(config, frameworks=("torch")):
    #  dqn = config.build()
    for i in range(70):
                result = dqn.train()
                print(
                    "Iteration {}, reward {}, timesteps {}".format(
                        i, result["episode_reward_mean"], result["timesteps_total"]
                    )
                ) 
    #checkpoint = dqn.save()
    #print("Checkpoint saved at", checkpoint)
    ray.shutdown()
