''' Batched Room-to-Room navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict, OrderedDict

import sys
sys.path.append('../build')

import MatterSim

from r2r.data_utils import load_nav_graphs
from r2r.data_utils import new_simulator
from r2r.data_utils import angle_feature, get_all_point_angle_feature

from r2r.eval_utils import cal_dtw, cal_cls

import h5py

ERROR_MARGIN = 3.0


class EnvBatch(object):
    ''' A simple wrapper for a batch of MatterSim environments,
        using discretized viewpoints and pretrained features '''

    def __init__(self, connectivity_dir, scan_data_dir=None, feat_db=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feat_db: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        self.feat_db = feat_db
        self.image_w = 640
        self.image_h = 480
        self.vfov = 60

        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            if scan_data_dir:
                sim.setDatasetPath(scan_data_dir)
            sim.setNavGraphPath(connectivity_dir)
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)

    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((36, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            feature = self.feat_db.get_image_feature(state.scanId, state.location.viewpointId)
            feature_states.append((feature, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input).
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)


class R2RBatch(object):
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(
        self, feat_db, instr_data, connectivity_dir,
        batch_size=64, angle_feat_size=4,
        seed=0, name=None, sel_data_idxs=None, args=None
    ):
        self.env = EnvBatch(connectivity_dir, feat_db=feat_db, batch_size=batch_size)

        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        # to evaluate full data
        self.gt_trajs = self._get_gt_trajs(self.data)

        # in validation, we would split the data
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]

        self.connectivity_dir = connectivity_dir
        self.angle_feat_size = angle_feat_size
        self.name = name
        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)

        self.dvae_probs = args.use_ig
        self.dvae_probs_dict = OrderedDict()

        self.args = args

        count = np.load("../visual_token_count.npy")
        count_dict = {}
        for i in range(len(count)):
            count_dict[i] = count[i]
        sort_count = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
        sort_index = [key for key, value in sort_count]
        self.sort_index = sort_index

        self.buffered_state_dict = {}
        #print(self.data)
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def _get_gt_trajs(self, data):
        return {x['instr_id']: (x['scan'], x['path']) for x in data}

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def get_dvae_probs(self, scan_id, viewpoint_id):
        key = '%s_%s' % (scan_id, viewpoint_id)
        if key in self.dvae_probs_dict:
            return self.dvae_probs_dict[key]
        else:
            with h5py.File(self.args.ig_path, 'r') as f:
                fts = f[key][...].astype(np.float32)
            if self.args.ig_head < 8192:
                fts = fts[:, self.sort_index[:self.args.ig_head]]
            self.dvae_probs_dict[key] = fts
            return fts

    def _next_minibatch(self, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        #print(len(self.data))
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.shortest_paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId, probs=None):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        # base_elevation = (viewId // 12 - 1) * math.radians(30)
        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)

        candidate_caption_dir = self.args.candidate_cap_dir + '/' + scanId + '/' + viewpointId + '/' + long_id + '.json'
        with open(candidate_caption_dir) as f:
            candidate_caption = json.load(f)

        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                long_id_ix = long_id+'_'+str(ix)

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]
                if self.dvae_probs:
                    prob = probs[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation, #new
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'caption': candidate_caption[long_id_ix], #new
                            'absolute_heading': state.heading, #new
                            'absolute_elevation': state.elevation #new
                         }
                        if self.dvae_probs:
                            adj_dict[loc.viewpointId]['ig_probs'] = prob
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'normalized_elevation', 'elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'caption', 'absolute_heading', 'absolute_elevation']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                c_new['elevation'] = c_new['normalized_elevation'] - 0
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                if self.dvae_probs:
                    prob = probs[ix]
                    c_new['ig_probs'] = prob
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def _teacher_path_action(self, state, path, t=None, shortest_teacher=False):
        if shortest_teacher:
            return self._shortest_path_action(state, path[-1])

        teacher_vp = None
        if t is not None:
            teacher_vp = path[t + 1] if t < len(path) - 1 else state.location.viewpointId
        else:
            if state.location.viewpointId in path:
                cur_idx = path.index(state.location.viewpointId)
                if cur_idx == len(path) - 1: # STOP
                    teacher_vp = state.location.viewpointId
                else:
                    teacher_vp = path[cur_idx + 1]
        return teacher_vp

    def _get_obs(self, t=None, shortest_teacher=False):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex

            if feature is None:
                feature = np.zeros((36, 2048))

            if self.dvae_probs:
                probs = self.get_dvae_probs(state.scanId, state.location.viewpointId)
            else:
                probs = None

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex, probs)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            #if self.args.llm_finetuning and self.args.chain_of_thought:
            obs.append({
                    'instr_id' : item['instr_id'],
                    'scan' : state.scanId,
                    'viewpoint' : state.location.viewpointId,
                    'viewIndex' : state.viewIndex,
                    'heading' : state.heading,
                    'elevation' : state.elevation,
                    'feature' : feature,
                    'candidate': candidate,
                    'navigableLocations' : state.navigableLocations,
                    'instruction' : item['instruction'],
                    'teacher' : self._teacher_path_action(state, item['path'], t=t, shortest_teacher=shortest_teacher),
                    'gt_path' : item['path'],
                    'path_id' : item['path_id'],
                })

            if self.dvae_probs:
                obs[-1]['ig_probs'] = probs
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.shortest_distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)

        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs(t=0)

    def step(self, actions, t=None):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs(t=t)


    ############### Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, scan, path, gt_path):
        scores = {}

        shortest_distances = self.shortest_distances[scan]

        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        nearest_position = self._get_nearest(shortest_distances, gt_path[-1], path)

        scores['nav_error'] = shortest_distances[path[-1]][gt_path[-1]]
        scores['oracle_error'] = shortest_distances[nearest_position][gt_path[-1]]
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])

        scores['success'] = float(scores['nav_error'] < ERROR_MARGIN)
        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['oracle_success'] = float(scores['oracle_error'] < ERROR_MARGIN)

        scores.update(
            cal_dtw(shortest_distances, path, gt_path, scores['success'], ERROR_MARGIN)
        )
        scores['CLS'] = cal_cls(shortest_distances, path, gt_path, ERROR_MARGIN)

        scores['ground_truth_steps'] = len(gt_path) - 1

        return scores

    def eval_metrics(self, preds, step_eval=False):
        ''' Evaluate each agent trajectory based on how close it got to the goal location
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        step_success = defaultdict(list)
        step_spl = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            traj = [x[0] for x in item['trajectory']]
            scan, gt_traj = self.gt_trajs[instr_id]
            # print(len(gt_traj)-1)
            traj_scores = self._eval_item(scan, traj, gt_traj)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
            step_success[traj_scores['ground_truth_steps']].append(traj_scores['success'])
            step_spl[traj_scores['ground_truth_steps']].append(traj_scores['spl'])

        avg_metrics = {
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'nDTW': np.mean(metrics['nDTW']) * 100,
            'SDTW': np.mean(metrics['SDTW']) * 100,
            'CLS': np.mean(metrics['CLS']) * 100,
        }

        if step_eval:
            step_sr_score = dict()
            step_spl_score = dict()
            print(step_success.keys())
            for k, v in step_success.items():
                step_sr_score[k] = np.mean(v)
                print(k, len(v))
                step_spl_score[k] = np.mean(step_spl[k])
            return avg_metrics, metrics, step_sr_score, step_spl_score
        else:
            return avg_metrics, metrics
