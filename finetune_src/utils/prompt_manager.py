
import string
import re
import random
from typing import Dict
import time
import json
import numpy as np

class PromptManager(object):
    def __init__(self, args):
        """
        1. Load nav_inputs
        2. Return prompt based on different modes
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.args = args
        self.history  = [[] for _ in range(self.args.batch_size)]
        self.results = []

    def make_action_options(self, cand_inputs, t):
        action_options_batch = []
        only_options_batch = []
        cand_action = cand_inputs["cand_action"]
        nav_types = cand_inputs["cand_nav_types"]
        batch_size = len(cand_action)

        for i in range(batch_size):
            actions = [action for nav_type, action in zip(nav_types[i],cand_action[i]) if nav_type == 1]

            if self.args.stop_first:
                actions =  ['stop'] + actions
            else:
                actions =  actions + ['stop']
            full_action_options = [chr(i + 65)+'. '+actions[i] for i in range(len(actions))]
            only_options = [chr(i + 65) for i in range(len(actions))]
            action_options_batch.append(full_action_options)
            only_options_batch.append(only_options)

        return action_options_batch, only_options_batch

    def make_history(self, a_t, nav_input, t):
        batch_size = len(a_t)
        for i in range(batch_size):

            if a_t[i] == -100:
                a_t[i] = len(nav_input["only_actions"][i])-1
            last_action = nav_input["only_actions"][i][a_t[i]]
            self.history[i] = 'Step ' + str(t+1) +". "+ last_action


    def get_prompt(self, obs, cand_inputs, mode, t):
        batch_size = len(obs)

        action_options_batch, only_options_batch = self.make_action_options(cand_inputs, t=t)
        prompt_batch = []

        if t == 0:
            example = 'Input: Instruction: walk towards the mirror, walk through the open door and stop. Observation: [A. stop, B. go forward <a wall with a mirror>, C. turn left <a bedroom with a bed>]. History: None.\nOutput: Imagination: mirror. Filtered observation: B matches the imagination. Action: B.\n'
        else:
            example = 'Input: Instruction: walk towards the mirror and walk through the open door. Observation: [A. stop,  B. go forward <a bedroom with a bed>, C. turn right <a open door leading to a hallway>]. History: Step 1. go forward <a wall with a mirror>.\nOutput: Imagination: open door. Filtered observation: C matches the imagination. Action: C.\n'

        for i in range(batch_size):
            instruction = obs[i]["instruction"]
            observation = None ###TODO
            if mode == 'navigation':

                observation_final = action_options_batch[i]
                if t == 0:
                    query = f"""Input: Instruction: {instruction} Observation: {observation_final}. History: None.\nOutput: """
                else:
                    query = f"""Input: Instruction: {instruction} Observation: {observation_final}. History: {self.history[i]}.\nOutput: """

                prompt = example + query

            prompt_batch.append(prompt)

        nav_input = {
            "prompts" : prompt_batch,
            "only_options": only_options_batch,
            "action_options": action_options_batch,
            "only_actions":  cand_inputs["cand_action"],
        }

        return nav_input

    def get_output(self, nav_output, only_options_batch, cand_inputs, t):
        batch_size = len(nav_output)
        output_batch = []
        output_index_batch = []

        for i in range(batch_size):

            output = nav_output[i].strip()
            #print(output)
            substr = "Action: "
            index = output.find(substr)
            if index == -1:
                output_index = random.randint(0, len(only_options_batch[i]) - 1)
                output_index_batch.append(output_index)
            else:
                option = output[index+8]

                if option in only_options_batch[i]:

                    output_batch.append(option)
                    output_index = only_options_batch[i].index(option)
                    output_index_batch.append(output_index)
                else:

                    output_index = random.randint(0, len(only_options_batch[i]) - 1)
                    output_index_batch.append(output_index)

        return output_index_batch





