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

import sys

import simpy
import random

from tdigest import TDigest
from enum import IntEnum
import itertools

RANDOM_SEED = 42

BOOT_TIME = 5      # Seconds it takes the tank truck to arrive
SIM_TIME = 110000000            # Simulation time in seconds
NUM_ARRIVALS = 10000
CONTROL_TIME_INTERVAL = 30.0
NUMBER_OF_NODES = 10
NUMBER_OF_PODS = 5
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

class Cluster(object):
    """A queue has a limited number of servers (``NUM_SERVERS``) to
    serve customers in parallel.

    customers have to request one of the servers. When they got one, they
    can start the serving processes and wait for it to finish (which
    takes ``washtime`` minutes).
    """
    def __init__(self, env: Any, NUMBER_OF_NODES: int,
                 number_of_servers: int) -> None:
        self.env = env
        self.number_of_nodes = NUMBER_OF_NODES
        self.number_of_pods = NUMBER_OF_PODS
        self.nodes = {i: simpy.Resource(self.env, self.number_of_nodes)
                      for i in range(self.number_of_hosts)}

        self.node_state = np.array([0 for _ in range(self.number_of_hosts)])
        self.node_state[0] = PowerState.ON

        self.digest = TDigest()     #  response time
        self.arrdigest = TDigest()  #  arrival time
        self.arrdigest.update(100.)
        self.digest.update(0.)

        self.active_nodes = 1

        # self.buffer=self.env.Store()
    def search_for_allocation(self) -> Any:
        lowest_load_node = self.number_of_nodes - 1
        for j in range(self.number_of_nodes):
            if self.node_state[j] == PowerState.ON:
                lowest_load_node = j
                break
        for i in range(j + 1, self.number_of_nodes):
            if (self.node_state[i] == PowerState.ON and
                    len(self.nodes[i].queue) + self.nodes[i].count <
                    len(self.nodes[lowest_load_node].queue) +
                    self.hosts[lowest_load_node].count):
                lowest_load_node = i

        assert lowest_load_node < self.number_of_hosts, \
            (f"node idx must be smaller than total number of nodes: {self.number_of_nodes} "f"expected, got: {lowest_load_node}")
        return lowest_load_node

    def search_for_off_host(self) -> Any:
        hostid = self.number_of_hosts
        for j in range(self.number_of_hosts):
            if self.host_state[j] == PowerState.ON_N:
                hostid = j
                #  print("Terminate the switch off")
                break
        if hostid == self.number_of_hosts:
            for i in range(self.number_of_hosts):
                if self.host_state[i] == PowerState.OFF:
                    hostid = i
                    break
        return hostid

    def scaleOut(self) -> Any:
        node_id = self.search_for_off_node()
        self.node_state[node_id] = ON



def customer(name, env, cluster):
    """A customer arrives at the cluster for service.
    It requests one of the servers. If there is no server available,
    the customer has to wait for a new server (takes time to boot a host.
    """
    #  print('No of active hosts %d' % cluster.active_num)
    #  print('%s arriving at cluster at %.1f' % (name, env.now))
    assignedto = cluster.search_for_allocation()
    #  print("Host %d queue size: %d " %
    #  (assignedto,len(cluster.hosts[assignedto].queue)))
    #  print("Being served:", cluster.hosts[assignedto].count)
    with cluster.hosts[assignedto].request() as req:
        start = env.now
        #  Request one of the servers from hosts[idh]
        yield req
        #  The "actual" service process takes some time
        t = random.expovariate(SERVICE_RATE)
        #  t=random.uniform(0.01,2)
        #  t=1
        yield env.timeout(t)
        #print('customer departsZZ')
        if (cluster.hosts[assignedto].count == 0 and
                cluster.host_state[assignedto] == PowerState.ON_N):
            cluster.host_state[assignedto] = PowerState.OFF
            cluster.active_num -= 1
        cluster.digest.update(env.now - start)
        #  print('No of active hosts %d' % cluster.active_num)
        #  print('%s finished service in %.7f seconds.
        #  service time %.7f' % (name,
        #  env.now - start,t))


def start_host(env, cluster, hostid):
    yield env.timeout(BOOT_TIME)
    #  print('Host %d ready at time %d' % (hostid,env.now))
    #  print('No of active hosts %d' % cluster.active_num)
    cluster.host_state[hostid] = PowerState.ON_A
    cluster.active_num += 1


def customer_generator(env, cluster):
    """ Generate new customers that arrive at the cluster."""
    for i in itertools.count():
        if env.now < SIM_TIME/2:
            t = random.expovariate(ARRIVAL_RATE)
        elif env.now < SIM_TIME*3/4:
            t = random.expovariate(ARRIVAL_RATE/2)
        else:
            t = random.expovariate(ARRIVAL_RATE/4)
        cluster.arrdigest.update(t)
        yield env.timeout(t)
        env.process(customer('Customer %d' % i, env, cluster,))

        if (i % 20000 ==0) :
           print('customer', i, "time --", env.now)

class ClusterEnv(ExternalEnv):
    def __init__(self, config: EnvContext):

        self.number_of_hosts = config["number_of_hosts"]
        self.num_of_pods = config["num_of_pods"]
        self.percentile_points = config["percentile_points"]
        self.number_of_active_hosts = 1
        self.observation_space = Tuple(
          [
             Box(0, np.inf, shape=(self.percentile_points,),
                            dtype=np.float64),
             #  Arrival cdf
             Box(0, np.inf, shape=(self.percentile_points,),
                            dtype=np.float64),
             #  response
             Box(0, 2, shape=(self.number_of_hosts,),
                            dtype=np.int64),
             #  state of hosts -- off, ON_A, ON_N
             Box(0, 10000, shape=(self.number_of_hosts,),
                            dtype=np.int64),
             #  number of customers at hosts
             #  Discrete(self.number_of_hosts),
             #  number of active hosts,  number of servers
             #  Discrete(self.number_of_host*self.capacity+1)
          ]
        )
        print("Initialization ---")
        #  print(self.observation_space)
        print("Initialization ---")
        self.action_space = Discrete(3)
        env_config = {
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }
        ExternalEnv.__init__(self, self.action_space, self.observation_space)
        self.simenv = simpy.Environment()
        self.cluster = Cluster(self.simenv, self.number_of_hosts,
                               self.num_servers)
        self.nc = 0
        self.arr = np.zeros(shape=(self.percentile_points,))
        self.ser = np.zeros(shape=(self.percentile_points,))
        self.numofcustomer = np.zeros(shape=(self.cluster.number_of_hosts,))
        for i in range(self.cluster.number_of_hosts):
            self.numofcustomer[i] = (len(self.cluster.hosts[i].queue) +
                                     self.cluster.hosts[i].count)
    #    self.reset(seed=config.worker_index * config.num_workers)

    def step(self, action):
        print("STEP?")
        return 100, 100, 100, {}

    def cluster_control(self):
        #  Periodically manages the cluster"""
        #  print("cluster control ---")
        yield self.simenv.timeout(CONTROL_TIME_INTERVAL)
        for i in range(self.percentile_points):
            self.arr[i] = self.cluster.arrdigest.percentile(i)
            self.ser[i] = self.cluster.digest.percentile(i)
        for i in range(self.cluster.number_of_hosts):
            self.numofcustomer[i] = (len(self.cluster.hosts[i].queue) +
                                     self.cluster.hosts[i].count)

        obs = tuple([self.arr, self.ser, self.cluster.host_state,
                     self.numofcustomer])
        while True:
            #  perform acion
            if(self.nc % 100 == 0) :
               print("cluster control ", self.nc, "time", self.simenv.now)
            self.nc +=1
            self.eid = self.start_episode()
            #  print(self.eid)
            self.action = self.action_space.sample()
            #  print(self.action)
            #  print("Obs: ", obs)
            #action = self.get_action(self.eid, obs)
            if self.action == Action.ScaleIn and self.cluster.active_num == 1:
                self.action == Action.Do_nothing

            if (self.action == Action.ScaleOut and
                    self.cluster.active_num == self.cluster.number_of_hosts):
                self.action == Action.Do_nothing

            self.log_action(self.eid, obs, self.action)
            #  print("Action %d" % self.action)
            #  print(obs)
            if self.action == Action.ScaleOut:
                hostid = self.cluster.search_for_off_host()
                #  print("scale out, host %d" %hostid)
                if hostid < self.cluster.number_of_hosts:
                    if self.cluster.host_state[hostid] == PowerState.OFF:
                        yield self.simenv.process(start_host(self.simenv,
                            self.cluster, hostid))
                    else:
                        self.cluster.host_state[hostid] = PowerState.ON_A
            elif self.action == Action.ScaleIn:
                hostid = self.cluster.search_for_allocation()
                #  print("scale in, host %d" % hostid)
                if (self.cluster.hosts[hostid].count == 0 and
                        len(self.cluster.hosts[hostid].queue) == 0):
                    self.cluster.host_state[hostid] = PowerState.OFF
                    self.cluster.active_num -= 1
                else:
                    self.cluster.host_state[hostid] = PowerState.ON_N
                    #  print('No of active hosts %d' % self.cluster.active_num)
            #  Check every 10 seconds
            #  reward here and observation
            del self.cluster.digest
            self.cluster.digest = TDigest()
            del self.cluster.arrdigest
            self.cluster.arrdigest = TDigest()

            yield self.simenv.timeout(CONTROL_TIME_INTERVAL)

            for i in range(self.percentile_points):
                self.arr[i] = self.cluster.arrdigest.percentile(i)
                self.ser[i] = self.cluster.digest.percentile(i)
            for i in range(self.cluster.number_of_hosts):
                self.numofcustomer[i] = (len(self.cluster.hosts[i].queue)
                                         + self.cluster.hosts[i].count)
                if self.numofcustomer[i] < 0:
                    print("ERROR")
            obs = tuple([self.arr, self.ser,
                         self.cluster.host_state, self.numofcustomer])
            excess_time = RESPONSE_TIME_THRESHOLD_U - self.cluster.digest.percentile(99.9)
            if (self.cluster.digest.percentile(99.9) >
                    RESPONSE_TIME_THRESHOLD_U):
                reward = -1 * math.log2(abs(excess_time))/2 * (self.action + 1)
            else:
                reward = excess_time * (self.action + 1) + 2

            #  compute reward, obs,info
            info = ""
            #  print(obs)
            #  print(reward)
            self.log_returns(self.eid, reward, info=info)
            self.end_episode(self.eid, obs)
            del self.cluster.digest
            self.cluster.digest = TDigest()
            del self.cluster.arrdigest
            self.cluster.arrdigest = TDigest()
            #  print('No of active hosts %d' % cluster.active_num)
            #  print("99.9 percentile %f" % cluster.digest.percentile(99.9))
            #  print("60 percentile %f" % cluster.digest.percentile(60))
            #  print("50 percentile %f" % cluster.digest.percentile(50))
            #  print("10 percentile %f" % cluster.digest.percentile(10))
            #  a consequence of  the previous action
            #  construct observation

    def run(self):
        #  print("Start -------")
        #  random.seed(RANDOM_SEED)
        #  np.array([(len(self.cluster.hosts[i].queue)+self.cluster.hosts[i].count
        #  for i in range(self.cluster.number_of_hosts))])
        #  self.hoststate=    numpy.array([self.cluster.host_state[i]
        #  for i in range(self.cluster.number_of_hosts)])
        self.simenv.process(self.cluster_control())
        self.simenv.process(customer_generator(self.simenv, self.cluster))
        print("Start sim...")
        self.simenv.run(until=SIM_TIME)
        print("sim vege...")

def env_creator(env_config):
     return ClusterEnv(env_config)



if __name__ == "__main__":

    random.seed(RANDOM_SEED)
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init(local_mode=args.local_mode,num_gpus=1)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config={
    #    "number_of_hosts": NUMBER_OF_HOSTS, "num_servers": NUMBER_OF_SERVERS, "percentile_points": 100}))
    
    #  register_env("cluster-env-v01", env_creator)
    register_env("cluster-env-v01", lambda _: ClusterEnv(config={
     "number_of_hosts": NUMBER_OF_HOSTS, "num_servers": NUMBER_OF_SERVERS, "percentile_points": 100}))


    config = (
            DQNConfig()
            .environment("cluster-env-v01")
            .rollouts(num_rollout_workers=1, enable_connectors=False)
    )
    dqn = config.build()
    #algo = dqn.DQN(env="cluster-env-v01")
    # policy_dict, is_policy_to_train = config.get_multi_agent_setup(env="cluster-env-v01")
    # is_policy_to_train("pol1")



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
