# !/usr/bin/python3
# -*- coding: utf-8 -*-

import gym
import copy
import numpy as np
from collections import deque

class FrameStack(gym.Wrapper):

    def __init__(self, env, k, DENOISE, type="RGB-frame"):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._perception_frames = deque([], maxlen=k)
        self.DENOISE = DENOISE
        self.type = type

        shp = env.observation_space.shape
        # import pdb; pdb.set_trace()

        if isinstance(shp, list) and len(shp) == 2:
            # Multi-Modals
            self.observation_space.shape = [
                (shp[0][0] * k,) + shp[0][1:],   # rgb
                (shp[1][0] * k,) + shp[1][1:],   # dvs
            ].copy()
        elif isinstance(shp, list) and len(shp) == 4:
            # Multi-Modals
            self.observation_space.shape = [
                (shp[0][0] * k,) + shp[0][1:],   # rgb
                (shp[1][0] * k,) + shp[1][1:],   # dvs-voxel
                # (shp[2][0] * k,) + shp[2][1:],   # depth
                shp[2],   # dvs-rec-frame
                (shp[3][0] * k,) + shp[3][1:],   # dvs-frame
            ].copy()
        elif shp[0] == 4:   # DVS-stream
            self.observation_space = gym.spaces.Box(
                low=0, high=1,  # not used
                shape=(4,),
                dtype=env.observation_space.dtype
            )
        else:               # Others
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=((shp[0] * k,) + shp[1:]),
                # shape=((shp[0][0] * k+shp[1][0]*k,) + shp[0][1:]+shp[1][1:]),
                dtype=env.observation_space.dtype
            )

    def _get_perception(self, obs):
        if self.perception_type == "RGB-frame":
            perception = obs[self.perception_type]
            return perception

        elif self.perception_type == "DVS-stream":
            perception = obs[self.perception_type]
            return perception

        elif self.perception_type == "DVS-frame":
            if self.DENOISE:
                perception = obs["Denoised-DVS-frame"][:, :, [0, 2]]
            else:
                perception = obs["DVS-frame"][:, :, [0, 2]]
            return perception

        elif self.perception_type == "DVS-voxel-grid":
            perception = obs[self.perception_type]
            return perception

        elif self.perception_type == "E2VID-frame":
            perception = obs[self.perception_type]
            return perception

        elif self.perception_type == "E2VID-latent":
            pass

        elif self.perception_type == "RGB-frame+DVS-frame":
            perception = [obs["RGB-frame"],
                          # obs["DVS-frame"][:, :, [0, 2]],
                          obs["Denoised-DVS-frame"][:, :, [0, 2]],
                          # obs["Depth-frame"],
                          ]
            return perception

        elif self.perception_type == "RGB-frame+DVS-voxel-grid":
            perception = [obs["RGB-frame"],
                          obs["DVS-voxel-grid"],
                          # obs["Depth-frame"],
                          # obs["events"],
                          # obs["DVS-frame"],
                          ]
            return perception


    def reset(self, selected_weather=None):
        obs = self.env.reset(selected_weather=selected_weather)
        for _ in range(self._k):
            self._perception_frames.append(self._get_perception(obs))

        stack_frames = self._get_stack_frames()
        obs.update({
            'perception': stack_frames
        })
        return obs


    def step(self, action):
        #         obs, reward, done, info = self.env.step(action)
        #         self._frames.append(obs)
        #         return self._get_obs(), reward, done, info
        obs, reward, done, info = self.env.step(action)
        # self._perception_frames.append(self._get_perception(obs))
        self._perception_frames.append(self._get_perception(obs))

        stack_frames = self._get_stack_frames()

        obs.update({
            'perception': stack_frames
        })
        return obs, reward, done, info

    def _create_dvs_rec_frame(self, dvs_events):
        """
        Create a frame from event data for reconstruction loss
        """
        dvs_events = np.concatenate(dvs_events, axis=0) # XYPT
        dvs_events = dvs_events[np.argsort(dvs_events[:, -1])]  
        t_start = dvs_events[0, -1]
        t_final = dvs_events[-1, -1]
        dt = t_final - t_start

        idx = 0
        data = []
        # Iterate over events for a window of dt
        while idx < dvs_events.shape[0]:
            e_curr = copy.deepcopy(dvs_events[idx])
            data.append(e_curr)

            if dt > 0:
                t_relative = float(t_final - e_curr[-1]) / dt
            else:
                t_relative = 0

            data[idx][-1] = t_relative

            idx += 1

        data = np.array(data)

        frame = np.zeros((self.observation_space.shape[2][1], self.observation_space.shape[2][2], 1), dtype=np.float32)

        for i in range(data.shape[0]):

            x_loc = int(data[i, 1])
            y_loc = int(data[i, 0])


            # Image contains (-1, 1) values if polarity is considered
            pix_val = data[i, 2]
            if pix_val == 0:    pix_val = -1

            # Scale polarity according to timestamp for a simple
            # representation to decode to
            pix_val = pix_val * data[i, 3]

            frame[x_loc, y_loc] += pix_val

        # Clip image representation to [-1, 1] for simplicity.
        dvs_rec_frame = np.clip(frame, -1, 1)

        dvs_rec_frame = np.transpose(dvs_rec_frame, (2, 0, 1))

        return dvs_rec_frame


    def _get_stack_frames(self):
        # assert len(self._perception_frames) == self._k

        if self.type.__contains__("+"):
            """
            RGB-frame+DVS-frame
            RGB-frame+DVS-voxel-grid
            """
            # 原本是(h,w,depth)
            rgb_frames, dvs_frames, dvs_events, dvs_frames2 = [], [], [], []
            for one_perception in self._perception_frames:
                rgb_frames.append(one_perception[0])
                dvs_frames.append(one_perception[1])

                # dvs_events.append(one_perception[2])    # events
                # dvs_frames2.append(one_perception[3])

            # dvs_rec_frame = self._create_dvs_rec_frame(dvs_events)

            # import pdb; pdb.set_trace()
            stack_frames = [
                np.transpose(np.concatenate(list(rgb_frames), axis=2), (2, 0, 1)).astype(np.float32),
                np.transpose(np.concatenate(list(dvs_frames), axis=2), (2, 0, 1)).astype(np.float32),
                # dvs_rec_frame.astype(np.float32),
                # np.transpose(np.concatenate(list(dvs_frames2), axis=2), (2, 0, 1)).astype(np.float32),
            ]

        else:
            if self.type.__contains__("stream"):
                # (event_num, 4) -> stack events within 3 frames
                stack_frames = np.concatenate(self._perception_frames, axis=0)
                stack_frames = stack_frames[np.argsort(stack_frames[:, -1])[::-1]]  # times is like: [1, ...., 0]

            else:
                stack_frames = np.concatenate(list(self._perception_frames), axis=2)
                stack_frames = np.transpose(stack_frames, (2, 0, 1)).astype(np.float32)

        # print("raw stack_frames.shape:", stack_frames.shape)
        # print("raw stack_frames.min:", stack_frames.min())
        # print("raw stack_frames.max:", stack_frames.max())
        # stack_frames = (np.transpose(stack_frames, (2, 0, 1)) / 255.).astype(np.float32)
        # print("af stack_frames.shape:", stack_frames.shape)
        # print("af stack_frames.min:", stack_frames.min())
        # print("af stack_frames.max:", stack_frames.max())
        return stack_frames

