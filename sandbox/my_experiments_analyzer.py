import sys
sys.path.append("ConvLab-2")
import importlib  
convlab2 = importlib.import_module("ConvLab-2.convlab2")
from convlab2.dialog_agent import BiSession
from convlab2.evaluator.multiwoz_eval import MultiWozEvaluator
from convlab2.nlg.template.multiwoz.nlg import TemplateNLG
from pprint import pprint
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import logging


class Analyzer:
    def __init__(self, user_agent, dataset='multiwoz'):
        self.user_agent = user_agent
        self.dataset = dataset
        self.usr_nlg = TemplateNLG(is_user=True)
        self.sys_nlg = TemplateNLG(is_user=False)

        assert self.user_agent.nlu is None
        assert self.user_agent.nlg is None

    def build_sess(self, sys_agent):
        if self.dataset == 'multiwoz':
            evaluator = MultiWozEvaluator()
        else:
            evaluator = None

        if evaluator is None:
            self.sess = None
        else:
            self.sess = BiSession(sys_agent=sys_agent, user_agent=self.user_agent, kb_query=None, evaluator=evaluator)
        return self.sess

    def dialog_analyze(self, sys_agent, total_dialog=100, dialog_length=40):
        max_session_turn = dialog_length
        sess = self.build_sess(sys_agent)

        goal_seeds = [random.randint(1,100000) for _ in range(total_dialog)]
        dialog_successes = []
        task_completes = []
        book_rates = []
        precisions = []
        recalls = []
        f1s = []
        usr_dialogs = [] #shape=(total_dialog, any), elem=[usr_utterance@1, usr_utterance@2, ..., usr_utterance@t, ...] 
        sys_dialogs = [] #shape=(total_dialog, any), elem=[sys_utterance@1, sys_utterance@2, ..., sys_utterance@t, ...]
        usr_dialog_acts = [] #shape=(total_dialog, any), elem=[usr_dialog_act@1, usr_dialog_act@2, ...]
        sys_dialog_acts = [] #shape=(total_dialog, any), elem=[sys_dialog_act@1, sys_dialog_act@2, ...]
        num_domains = [] # shape = (total_dialog, ), elem = [num_domain of dialog]
        num_domains_satisfying_constraints = [] # shape = (total_dialog, ), elem = [num_domain satisfying goal constraints]
        satisfying_constraints = [] # shape=(total_dialog, ), elem=[1 or 0] where 1 means satisfying and 0 means not satisfying.
        

        for j in tqdm(range(total_dialog), desc="dialogue"):
            sys_response = '' if self.user_agent.nlu else []
            random.seed(goal_seeds[0])
            np.random.seed(goal_seeds[0])
            torch.manual_seed(goal_seeds[0])
            goal_seeds.pop(0)
            sess.init_session()

            usr_dialog_acts_tmp = []
            sys_dialog_acts_tmp = []
            usr_utterances = []
            sys_utterances = []

            for i in range(max_session_turn):
                sys_response, user_response, session_over, reward = sess.next_turn(sys_response)
                sys_dialog_acts_tmp.append(user_response)
                usr_dialog_acts_tmp.append(sys_response)
                usr_utterances.append(self.usr_nlg.generate(user_response))
                sys_utterances.append(self.sys_nlg.generate(sys_response))

                if session_over:
                    break

            task_success = sess.evaluator.task_success()
            task_complete = sess.user_agent.policy.policy.goal.task_complete()
            book_rate = sess.evaluator.book_rate()
            stats = sess.evaluator.inform_F1() # (prec, rec, F1): prec: systemが与えた全情報のうち要求されていた情報の割合，rec: userが要求した全情報の内，systemが与えた情報の割合
            percentage = sess.evaluator.final_goal_analyze()

            dialog_successes.append(task_success)
            task_completes.append(task_complete)
            book_rates.append(book_rate)
            precisions.append(stats[0])
            recalls.append(stats[1])
            f1s.append(stats[2])
            num_domains.append(len(sess.evaluator.goal))
            num_domains_satisfying_constraints.append(len(sess.evaluator.goal)*percentage)
            satisfying_constraints.append(percentage==1)
            usr_dialog_acts.append(usr_dialog_acts_tmp)
            sys_dialog_acts.append(sys_dialog_acts_tmp)
            usr_dialogs.append(usr_utterances)
            sys_dialogs.append(sys_utterances)
        
        return (dialog_successes,
                task_completes,
                book_rates,
                precisions,
                recalls,
                f1s,
                num_domains,
                num_domains_satisfying_constraints,
                satisfying_constraints,
                usr_dialog_acts,
                sys_dialog_acts,
                usr_dialogs,
                sys_dialogs)
