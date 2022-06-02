from shutil import unregister_archive_format
import sys
sys.path.append("convlab2")
import importlib  
convlab2 = importlib.import_module("convlab2")
from convlab2.dialog_agent import BiSession
from convlab2.nlg.template.multiwoz.nlg import TemplateNLG
from convlab2.dialog_agent.env import Environment
from convlab2.policy.rule.multiwoz import RulePolicy
from convlab2.dialog_agent.agent import PipelineAgent
from convlab2.dst.rule.multiwoz import RuleDST
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.policy.ppo import PPO
from convlab2.policy.gdpl import GDPL, RewardEstimator
import random
import numpy as np
from pprint import pprint
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from typing import List
from argparse import ArgumentParser
from my_experiments_analyzer import Analyzer


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


def sampler(env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 50
    real_traj_len = 0

    sessions = []

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

            # save 1 turn
            sessions.append((s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask))

            # update per step
            s = next_s
            real_traj_len = t

            print(f"t: {t}, r: {r:.3f}")

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length
    
    return sessions



def sample(env, policy, batchsz):
    """
    Given batchsz number of task, the batchsz will be splited equally to each processes
    and when processes return, it merge all data and return
	:param env:
	:param policy:
    :param batchsz:
    :return: batch
    """

    samples = sampler(env, policy, batchsz)
    # (s_vec.numpy(), policy.vector.action_vectorize(a), r, next_s_vec.numpy(), mask)
    return Batch(
                [e[0] for e in samples],
                [e[1] for e in samples],
                [e[2] for e in samples],
                [e[3] for e in samples],
                [e[4] for e in samples]
    )

def update(env, policy, batchsz, epoch, rewarder=None):
    # sample data asynchronously
    batch = sample(env, policy, batchsz)

    # data in batch is : batch.state: ([1, s_dim], [1, s_dim]...)
    # batch.action: ([1, a_dim], [1, a_dim]...)
    # batch.reward/ batch.mask: ([1], [1]...)
    s = torch.from_numpy(np.stack(batch.states)).to(device=DEVICE)
    a = torch.from_numpy(np.stack(batch.actions)).to(device=DEVICE)
    r = torch.from_numpy(np.stack(batch.rewards)).to(device=DEVICE)
    next_s = torch.from_numpy(np.stack(batch.next_states)).to(device=DEVICE)
    mask = torch.Tensor(np.stack(batch.masks)).to(device=DEVICE)
    batchsz_real = s.size(0)

    if rewarder is None:
        policy.update(epoch, batchsz_real, s, a, r, mask)
    else:
        policy.update(epoch, batchsz_real, s, a, next_s, mask, rewarder)

def generate_example_dialog():
    # create user agent
    policy_usr = RulePolicy(character='usr')
    nlg_usr = TemplateNLG(is_user=True)
    user_agent = PipelineAgent(nlu=None, dst=None, policy=policy_usr, nlg=None, name='user')

    # create system agent
    dst_sys = RuleDST()
    nlg_sys = TemplateNLG(is_user=False)
    policy_sys = GDPL(is_train=False)
    sys_agent = PipelineAgent(nlu=None, dst=dst_sys, policy=policy_sys, nlg=None, name='sys')

    # create evaluator
    evaluator = MultiWozEvaluator()

    sess = BiSession(sys_agent, user_agent, kb_query=None, evaluator=evaluator)
    
    set_seed(100)

    sys_response = []
    sess.init_session()
    print('initial goal:')
    pprint(sess.evaluator.goal)
    print('-'*50)
    for i in range(20):
        sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
        print(f'user: {nlg_usr.generate(user_response)}')
        print(f'system: {nlg_sys.generate(sys_response)}')
        print()
        if session_over is True:
            break
    print(f'task success: {sess.evaluator.task_success()}')
    print(f'task complete: {sess.user_agent.policy.policy.goal.task_complete()}')
    print(f'book_rate: {sess.evaluator.book_rate()}')
    print(f'inform F1: {sess.evaluator.inform_F1()}') # (prec, rec, F1): prec: systemが与えた全情報のうち要求されていた情報の割合，rec: userが要求した全情報の内，systemが与えた情報の割合
    print('-'*50)
    print('final goal')
    pprint(sess.evaluator.goal)
    print('='*100)


def check_analyzer():
    # create user agent
    policy_usr = RulePolicy(character='usr')
    user_agent = PipelineAgent(nlu=None, dst=None, policy=policy_usr, nlg=None, name='user')

    # create system agent
    dst_sys = RuleDST()
    policy_sys = GDPL(is_train=False)
    sys_agent = PipelineAgent(nlu=None, dst=dst_sys, policy=policy_sys, nlg=None, name='sys')

    set_seed(20220501)
    analyzer = Analyzer(user_agent)
    print(analyzer.dialog_analyze(sys_agent, total_dialog=100, dialog_length=20)[6])



def main(args):
    # create user simulator
    policy_usr = RulePolicy(character='usr')
    simulator = PipelineAgent(None, None, policy_usr, None, 'user')

    # create dst
    dst_sys = RuleDST()

    # create evaluator
    evaluator = MultiWozEvaluator()
    
    # create env
    env = Environment(None, simulator, None, dst_sys, evaluator)

    # create RL's policy
    policy_sys = GDPL(is_train=True)
    rewarder = RewardEstimator(policy_sys.vector, False)

    #for i in range(args.epoch):
    #    update(env, policy_sys, args.batchsz, i, rewarder)

    # plot example of dialogue
    sys_agent = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')
    evaluate(sys_agent, simulator, evaluator)

    # analysis the result
    #sys_agent = PipelineAgent(None, dst_sys, policy_sys, None, 'sys')
    # if sys_nlu!=None, set use_nlu=True to collect more information
    #analyzer = Analyzer(user_agent=simulator, dataset='multiwoz')

    #set_seed(20200131)
    #analyzer.comprehensive_analyze(sys_agent=sys_agent, model_name='sys_agent', total_dialog=100)



if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--batchsz", type=int, default=1024, help="batch size of trajactory sampling")
    parser.add_argument("--epoch", type=int, default=1, help="number of epochs to train")
    args = parser.parse_args()
    import time
    st = time.time()
    #main(args)
    #generate_example_dialog()
    check_analyzer()
    print(f"elapsed {(time.time() - st):.3f}[sec]")
