import logging
logging.basicConfig(level=logging.WARNING)
import sys
sys.path.append("ConvLab-2")
import importlib  
convlab2 = importlib.import_module("ConvLab-2.convlab2")
from convlab2.dialog_agent.env import Environment
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.policy.rlmodule import Memory, Transition
from convlab2.policy.ppo import PPO
from convlab2.policy.gdpl import GDPL, RewardEstimator
from convlab2.util.analysis_tool.analyzer import Analyzer
import random
import numpy as np
import torch
from torch import multiprocessing as mp
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import List
from argparse import ArgumentParser
import mypy_extensions as mx


class Batch():
    def __init__(self, states, actions, rewards, next_states, masks):
        self.states: List[List[float]]= states
        self.actions: List[List[float]] = actions
        self.rewards: List[List[int]] = rewards
        self.next_states: List[List[float]] = next_states
        self.masks: List[List[int]] = masks


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()
    last_rewards = []

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(policy.vector.state_vectorize(s))
            a = policy.predict(s)

            # interact with env
            next_s, r, done = env.step(a)

            # a flag indicates ending or not
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(policy.vector.state_vectorize(next_s))

            # save to queue
            buff.push(s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                last_rewards.append(r)
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    assert sampled_traj_num == len(last_rewards)
    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff, last_rewards])
    evt.wait()


def sample(env, policy, batchsz, process_num):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
	:param process_num:
    :return: batch
    """

    # batchsz will be splitted into each process,
    # final batchsz maybe larger than batchsz parameters
    process_batchsz = np.ceil(batchsz / process_num).astype(np.int32)
    # buffer to save all data
    queue = mp.Queue()

    # start processes for pid in range(1, processnum)
    # if processnum = 1, this part will be ignored.
    # when save tensor in Queue, the process should keep alive till Queue.get(),
    # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
    # however still some problem on CUDA tensors on multiprocessing queue,
    # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
    # so just transform tensors into numpy, then put them into queue.
    evt = mp.Event()
    processes = []
    for i in range(process_num):
        process_args = (i, queue, evt, env, policy, process_batchsz)
        processes.append(mp.Process(target=sampler, args=process_args))
    for p in processes:
        # set the process as daemon, and it will be killed once the main process is stoped.
        p.daemon = True
        p.start()

    # we need to get the first Memory object and then merge others Memory use its append function.
    pid0, buff0, last_rewards0 = queue.get()
    for _ in range(1, process_num):
        pid, buff_, last_rewards = queue.get()
        buff0.append(buff_)  # merge current Memory into buff0
        last_rewards0.extend(last_rewards)
    evt.set()

    # now buff saves all the sampled data
    buff = buff0
    last_rewards = last_rewards0

    return buff.get_batch(), last_rewards


def update(env, policy, batchsz, epoch, process_num, rewarder=None):
    # sample data asynchronously
    batch, last_rewards = sample(env, policy, batchsz, process_num)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    batchsz_real = s.size(0)

    if rewarder is None:
        policy.update(epoch, batchsz_real, s, a, r, mask)
    else:
        policy.update(epoch, batchsz_real, s, a, next_s, mask, rewarder)
    
    return np.mean(last_rewards), np.std(last_rewards)



def main(args):
    # create user simulator
    policy_usr = RulePolicy(character='usr')
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    # create dst
    dst_sys = RuleDST()

    # create evaluator
    evaluator = MultiWozEvaluator()
    
    # create env
    env = Environment(None, simulator, None, dst_sys, evaluator=None)

    # create RL's policy
    policy_sys = GDPL(is_train=True)
    rewarder = RewardEstimator(policy_sys.vector, False)

    last_reward_stats = [] # List[(mean, std)]
    for i in range(args.epoch):
        last_reward_mean, last_reward_std = update(env, policy_sys, args.batchsz, i, args.process_num, rewarder)
        last_reward_stats.append((last_reward_mean, last_reward_std))

    print(last_reward_stats)

    # analysis the result
    #sys_agent = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')
    # if sys_nlu!=None, set use_nlu=True to collect more information
    #analyzer = Analyzer(user_agent=simulator, dataset='multiwoz')

    #set_seed(20200131)
    #analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='sys_agent', total_dialog=100)



if __name__=='__main__':
    import os
    print(os.cpu_count())
    parser = ArgumentParser()
    parser.add_argument("--batchsz", type=int, default=1024, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=30, help="number of epochs to train")
    parser.add_argument("--process_num", type=int, default=8, help="number of processes of trajactory sampling")
    args = parser.parse_args()
    import time
    st = time.time()
    main(args)
    print(f"elapsed {(time.time() - st):.3f}[sec]")
