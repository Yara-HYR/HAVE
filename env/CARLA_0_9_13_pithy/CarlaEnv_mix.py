# !/usr/bin/python3
# -*- coding: utf-8 -*-
# CARLA 0.9.13 environment

"""
sudo docker run --privileged --user carla --gpus all --net=host -e DISPLAY=$DISPLAY carlasim/carla-add:0.9.13 /bin/bash ./CarlaUE4.sh -world-port=12321 -RenderOffScreen
"""

import carla

import os
import sys
import time
import math
import torch
import random
import numpy as np
from copy import deepcopy
from dotmap import DotMap

from denoise.event_process import dist_main_noise
from denoise.Contrast_Maximization import contra_max

class CarlaEnv(object):

    def __init__(self,
                 weather_params, scenario_params,
                 selected_weather, selected_scenario,
                 carla_rpc_port, carla_tm_port, carla_timeout,
                 perception_type, num_cameras, rl_image_size, fov,
                 max_fps, min_fps, device,
                 min_stuck_steps, max_episode_steps, frame_skip,
                 DENOISE=False, is_spectator=False, ego_auto_pilot=False,
                 dvs_rec_args=None, TPV=False, BEV=False
                 ):

        self.device = device
        self.DENOISE = DENOISE
        self.TPV = TPV
        self.BEV = BEV
        self.frame_skip = frame_skip

        self.carla_rpc_port = carla_rpc_port
        self.carla_tm_port = carla_tm_port
        self.carla_timeout = carla_timeout
        self.weather_params = weather_params
        self.scenario_params = scenario_params

        # testing params
        self.ego_auto_pilot = ego_auto_pilot
        self.is_spectator = is_spectator

        self.num_cameras = num_cameras
        self.rl_image_size = rl_image_size
        self.fov = fov
        self.max_fps = max_fps
        self.min_fps = min_fps
        if max_episode_steps is None:
            self.max_episode_steps = 20 * self.max_fps
        else:
            self.max_episode_steps = max_episode_steps
        if min_stuck_steps is None:
            self.min_stuck_steps = 2 * self.max_fps
        else:
            self.min_stuck_steps = min_stuck_steps

        self.selected_weather = selected_weather
        self.selected_scenario = selected_scenario

        # rgb-frame, dvs-rec-frame, dvs-stream, dvs-vidar-stream
        self.perception_type = perception_type  # ↑↑↑↑↑↑↑↑↑
        if self.perception_type.__contains__("E2VID"):
            assert dvs_rec_args, "missing necessary param: [dvs_rec_args]"

            self.dvs_rec_args = dvs_rec_args

            sys.path.append("./tools/rpg_e2vid")
            # from tools.rpg_e2vid.run_dvs_rec import run_dvs_rec
            # from run_dvs_rec import run_dvs_rec
            # from tools.rpg_e2vid.e2vid_utils.loading_utils import load_model, get_device
            from e2vid_utils.loading_utils import load_model, get_device

            # Load model
            self.rec_model = load_model(self.dvs_rec_args.path_to_model)
            self.device = get_device(self.dvs_rec_args.use_gpu)
            self.rec_model = self.rec_model.to(self.device)
            self.rec_model .eval()

        elif self.perception_type == "vidar-rec-frame":
            sys.path.append("./tools/vidar2frame")
            # vidar reconstruction do not need to load a model


        # client init
        self.client = carla.Client('localhost', self.carla_rpc_port)
        self.client.set_timeout(self.carla_timeout)

        # world
        self.world = self.client.load_world(self.scenario_params[self.selected_scenario]["map"])

        # assert self.client.get_client_version() == "0.9.13"
        # assert self.selected_scenario in self.scenario_params.keys()
        # print(self.weather_params.keys())
        # print(self.selected_weather)
        # assert self.selected_weather in self.weather_params.keys()

        #
        self.vehicle_actors = []
        self.sensor_actors = []
        self.walker_ai_actors = []
        self.walker_actors = []

        self.reset_num = 0

        # reset
        self.reset()

    def _init_blueprints(self):

        self.bp_lib = self.world.get_blueprint_library()

        self.collision_bp = self.bp_lib.find('sensor.other.collision')

        self.bev_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.bev_camera_bp.set_attribute('sensor_tick', f'{1 / self.min_fps}')
        self.bev_camera_bp.set_attribute('image_size_x', str(2048))
        self.bev_camera_bp.set_attribute('image_size_y', str(2048))
        self.bev_camera_bp.set_attribute('fov', str(90))
        self.bev_camera_bp.set_attribute('enable_postprocess_effects', str(True))

        self.video_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.video_camera_bp.set_attribute('sensor_tick', f'{1 / self.min_fps}')
        self.video_camera_bp.set_attribute('image_size_x', str(1024))
        self.video_camera_bp.set_attribute('image_size_y', str(1024))

        self.rgb_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('sensor_tick', f'{1 / self.min_fps}')
        self.rgb_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.rgb_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        self.rgb_camera_bp.set_attribute('fov', str(self.fov))
        self.rgb_camera_bp.set_attribute('enable_postprocess_effects', str(True))   # a set of post-process effects is applied to the image to create a more realistic feel
        # self.rgb_camera_bp.set_attribute('blur_amount', '1.0')
        # self.rgb_camera_bp.set_attribute('motion_blur_intensity', '1.0')
        # self.rgb_camera_bp.set_attribute('motion_blur_max_distortion', '0.8')
        # self.rgb_camera_bp.set_attribute('motion_blur_min_object_screen_size', '0.4')
        self.rgb_camera_bp.set_attribute('exposure_max_bright', '15.0')     # over-exposure
        self.rgb_camera_bp.set_attribute('exposure_min_bright', '12.0')      # under-exposure
        self.rgb_camera_bp.set_attribute('blur_amount', '1.0')
        self.rgb_camera_bp.set_attribute('motion_blur_intensity', '1.0')
        self.rgb_camera_bp.set_attribute('motion_blur_max_distortion', '0.8')
        self.rgb_camera_bp.set_attribute('motion_blur_min_object_screen_size', '0.4')
        self.rgb_camera_bp.set_attribute('exposure_speed_up', '3.0')    # Speed at which the adaptation occurs from dark to bright environment.
        self.rgb_camera_bp.set_attribute('exposure_speed_down', '1.0')  # Speed at which the adaptation occurs from bright to dark environment.
        self.rgb_camera_bp.set_attribute('lens_flare_intensity', '0.2')  #  Intensity for the lens flare post-process effect （光晕效果）
        self.rgb_camera_bp.set_attribute('shutter_speed', '100')  # The camera shutter speed in seconds 快门速度


        self.dvs_camera_bp = self.bp_lib.find('sensor.camera.dvs')
        self.dvs_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
        self.dvs_camera_bp.set_attribute('positive_threshold', str(0.3))
        self.dvs_camera_bp.set_attribute('negative_threshold', str(0.3))
        self.dvs_camera_bp.set_attribute('sigma_positive_threshold', str(0.05))
        self.dvs_camera_bp.set_attribute('sigma_negative_threshold', str(0.05))
        self.dvs_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.dvs_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        # self.dvs_camera_bp.set_attribute('use_log', str(False))
        self.dvs_camera_bp.set_attribute('use_log', str(True))
        self.dvs_camera_bp.set_attribute('fov', str(self.fov))
        self.dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))

        # self.dvs_camera_bp = self.bp_lib.find('sensor.camera.dvs')
        # self.dvs_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
        # #         dvs_camera_bp.set_attribute('positive_threshold', str(0.3))
        # #         dvs_camera_bp.set_attribute('negative_threshold', str(0.3))
        # #         dvs_camera_bp.set_attribute('sigma_positive_threshold', str(0))
        # #         dvs_camera_bp.set_attribute('sigma_negative_threshold', str(0))
        # self.dvs_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        # self.dvs_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        # self.dvs_camera_bp.set_attribute('fov', str(self.fov))
        # self.dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))

        # self.depth_camera_bp = self.bp_lib.find('sensor.camera.depth')
        # self.depth_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        # self.depth_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        # self.dvs_camera_bp.set_attribute('fov', str(self.fov))
        # self.dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))

        self.depth_camera_bp = self.bp_lib.find('sensor.camera.depth')
        self.depth_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.depth_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        self.dvs_camera_bp.set_attribute('fov', str(self.fov))
        self.dvs_camera_bp.set_attribute('enable_postprocess_effects', str(True))


        self.vidar_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.vidar_camera_bp.set_attribute('sensor_tick', f'{1 / self.max_fps}')
        self.vidar_camera_bp.set_attribute('image_size_x', str(self.rl_image_size))
        self.vidar_camera_bp.set_attribute('image_size_y', str(self.rl_image_size))
        self.vidar_camera_bp.set_attribute('fov', str(self.fov))
        self.vidar_camera_bp.set_attribute('enable_postprocess_effects', str(True))

    def _set_dummy_variables(self):
        # dummy variables given bisim's assumption on deep-mind-control suite APIs
        low = -1.0
        high = 1.0
        self.action_space = DotMap()
        self.action_space.low.min = lambda: low
        self.action_space.high.max = lambda: high
        self.action_space.shape = [2]
        self.observation_space = DotMap()
        # D, H, W
        # before stack
        if self.perception_type == "RGB-frame":
            self.observation_space.shape = (3, self.rl_image_size, self.num_cameras * self.rl_image_size)
            self.observation_space.dtype = np.dtype(np.float32)
        elif self.perception_type == "DVS-frame":
            self.observation_space.shape = (2, self.rl_image_size, self.num_cameras * self.rl_image_size)
            self.observation_space.dtype = np.dtype(np.float32)
        elif self.perception_type == "DVS-voxel-grid":
            self.observation_space.shape = (5, self.rl_image_size, self.rl_image_size * self.num_cameras)
            self.observation_space.dtype = np.dtype(np.float32)
        elif self.perception_type == "E2VID-frame":
            self.observation_space.shape = (1, self.rl_image_size, self.num_cameras * self.rl_image_size)
            self.observation_space.dtype = np.dtype(np.float32)
        elif self.perception_type == "E2VID-latent":
            self.observation_space.shape = (1, self.rl_image_size, self.num_cameras * self.rl_image_size)
            self.observation_space.dtype = np.dtype(np.float32)
        elif self.perception_type == "RGB-frame+DVS-frame":
            self.observation_space.shape = [
                (3, self.rl_image_size, self.num_cameras * self.rl_image_size),
                (2, self.rl_image_size, self.num_cameras * self.rl_image_size),
                # (1, self.rl_image_size, self.num_cameras * self.rl_image_size),
                # (3, self.rl_image_size, self.num_cameras * self.rl_image_size),
            ]
            self.observation_space.dtype = np.dtype(np.float32)
        elif self.perception_type == "RGB-frame+DVS-voxel-grid":
            self.observation_space.shape = [
                (3, self.rl_image_size, self.num_cameras * self.rl_image_size),
                (5, self.rl_image_size, self.num_cameras * self.rl_image_size), # DVS-voxel-grid
                # (1, self.rl_image_size, self.num_cameras * self.rl_image_size),
                # (3, self.rl_image_size, self.num_cameras * self.rl_image_size), # DVS-frame
            ]
            self.observation_space.dtype = np.dtype(np.float32)

        # if self.perception_type.__contains__("stream"):
        #     self.observation_space.shape = (None, 4)
        #     self.observation_space.dtype = np.dtype(np.float32)
        self.reward_range = None
        self.metadata = None
        self.action_space.sample = lambda: np.random.uniform(
            low=low, high=high, size=self.action_space.shape[0]).astype(np.float32)

    def _dist_from_center_lane(self, vehicle, info):
        # assume on highway
        vehicle_location = vehicle.get_location()
        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
        vehicle_xy = np.array([vehicle_location.x, vehicle_location.y])
        vehicle_s = vehicle_waypoint.s
        vehicle_velocity = vehicle.get_velocity()  # Vecor3D
        vehicle_velocity_xy = np.array([vehicle_velocity.x, vehicle_velocity.y])
        speed = np.linalg.norm(vehicle_velocity_xy)

        vehicle_waypoint_closest_to_road = \
            self.map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        road_id = vehicle_waypoint_closest_to_road.road_id
        assert road_id is not None
        lane_id = int(vehicle_waypoint_closest_to_road.lane_id)
        goal_lane_id = lane_id

        current_waypoint = self.map.get_waypoint(vehicle_location, project_to_road=False)
        goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
        if goal_waypoint is None:
            # try to fix, bit of a hack, with CARLA waypoint discretizations
            carla_waypoint_discretization = 0.02  # meters
            goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s - carla_waypoint_discretization)
            if goal_waypoint is None:
                goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id,
                                                           vehicle_s + carla_waypoint_discretization)

        if goal_waypoint is None:
            # print("Episode fail: goal waypoint is off the road! (frame %d)" % self.time_step)
            done, dist, vel_s = True, 100., 0.
            info['reason_episode_ended'] = 'off_road'

        else:
            goal_location = goal_waypoint.transform.location
            goal_xy = np.array([goal_location.x, goal_location.y])
            dist = np.linalg.norm(vehicle_xy - goal_xy)

            next_goal_waypoint = goal_waypoint.next(0.1)  # waypoints are ever 0.02 meters
            if len(next_goal_waypoint) != 1:
                print('warning: {} waypoints (not 1)'.format(len(next_goal_waypoint)))
            if len(next_goal_waypoint) == 0:
                print("Episode done: no more waypoints left. (frame %d)" % self.time_step)
                info['reason_episode_ended'] = 'no_waypoints'
                done, vel_s = True, 0.
            else:
                location_ahead = next_goal_waypoint[0].transform.location
                highway_vector = np.array([location_ahead.x, location_ahead.y]) - goal_xy
                highway_unit_vector = np.array(highway_vector) / np.linalg.norm(highway_vector)
                vel_s = np.dot(vehicle_velocity_xy, highway_unit_vector)
                done = False

        # not algorithm's fault, but the simulator sometimes throws the car in the air wierdly
        if vehicle_velocity.z > 1. and self.time_step < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_velocity.z, self.time_step))
            info['reason_episode_ended'] = 'carla_bug'
            done = True
        if vehicle_location.z > 0.5 and self.time_step < 20:
            print("Episode done: vertical velocity too high ({}), usually a simulator glitch (frame {})".format(vehicle_location.z, self.time_step))
            info['reason_episode_ended'] = 'carla_bug'
            done = True

        return dist, vel_s, speed, done

    def _on_collision(self, event):
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        # print('Collision (intensity {})'.format(intensity))
        self._collision_intensities_during_last_time_step.append(intensity)

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.

        Args:
            filt: the filter indicating what type of actors we'll look at.

        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly

        return actor_poly_dict

    def _control_all_walkers(self):

        walker_behavior_params = self.scenario_params[self.selected_scenario]["walker_behavior"]

        # if walker is dead 
        # for walker in self.walker_actors:
        #     if not walker.is_alive:
        #         walker.destroy()
        #         self.walker_actors.remove(walker)

        # all_veh_locs = [
        #     [one_actor.get_transform().location.x, one_actor.get_transform().location.y]
        #     for one_actor in self.vehicle_actors
        # ]
        # all_veh_locs = np.array(all_veh_locs, dtype=np.float32)

        for walker in self.walker_actors:
            if walker.is_alive:
                # get location and velocity of the walker
                loc_x, loc_y = walker.get_location().x, walker.get_location().y
                vel_x, vel_y = walker.get_velocity().x, walker.get_velocity().y
                walker_loc = np.array([loc_x, loc_y], dtype=np.float32)

                # judge whether walker can cross the road
                # dis_gaps = np.linalg.norm(all_veh_locs - walker_loc, axis=1)
                # cross_flag = (dis_gaps >= walker_behavior_params["secure_dis"]).all()
                cross_prob = walker_behavior_params["cross_prob"]

                if loc_y > walker_behavior_params["border"]["y"][1]:
                    if self.time_step % self.max_fps == 0 and random.random() < cross_prob:
                        walker.apply_control(self.left)
                    # else:
                    #     if loc_x > walker_behavior_params["border"]["x"][1]:
                    #         walker.apply_control(self.backward)
                    #
                    #     elif loc_x > walker_behavior_params["border"]["x"][0]:
                    #         if vel_x > 0:
                    #             walker.apply_control(self.forward)
                    #         else:
                    #             walker.apply_control(self.backward)
                    #
                    #     else:
                    #         walker.apply_control(self.forward)

                elif loc_y > walker_behavior_params["border"]["y"][0]:
                    if vel_y > 0:
                        walker.apply_control(self.right)
                    else:
                        walker.apply_control(self.left)

                else:
                    if self.time_step % self.max_fps == 0 and random.random() < cross_prob:
                        walker.apply_control(self.right)
                    #
                    # else:
                    #     if loc_x > walker_behavior_params["border"]["x"][1]:
                    #         walker.apply_control(self.backward)
                    #
                    #     elif loc_x > walker_behavior_params["border"]["x"][0]:
                    #         if vel_x > 0:
                    #             walker.apply_control(self.forward)
                    #         else:
                    #             walker.apply_control(self.backward)
                    #
                    #     else:
                    #         walker.apply_control(self.forward)

    def _clear_all_actors(self):
        # remove all vehicles, walkers, and sensors (in case they survived)
        # self.world.tick()

        if 'vehicle' in dir(self) and self.vehicle is not None:
            for one_sensor_actor in self.sensor_actors:
                if one_sensor_actor.is_alive:
                    one_sensor_actor.stop()
                    one_sensor_actor.destroy()

        # # self.vidar_data['voltage'] = np.zeros((self.obs_size, self.obs_size), dtype=np.uint16)
        for actor_filter in ['vehicle.*', 'walker.*']:
            for actor in self.world.get_actors().filter(actor_filter):
                if actor.is_alive:
                    actor.destroy()

        # for one_vehicle_actor in self.vehicle_actors:
        #     if one_vehicle_actor.is_alive:
        #         one_vehicle_actor.destroy()

        # for one_walker_ai_actor in self.walker_ai_actors:
        #     if one_walker_ai_actor.is_alive:
        #         one_walker_ai_actor.stop()
        #         one_walker_ai_actor.destroy()

        # for one_walker_actor in self.walker_actors:
        #     if one_walker_actor.is_alive:
        #         one_walker_actor.destroy()


        # for actor_filter in ['vehicle.*', 'controller.ai.walker', 'walker.*', 'sensor*']:
        #     for actor in self.world.get_actors().filter(actor_filter):
        #         if actor.is_alive:
        #             if actor.type_id == 'controller.ai.walker':
        #                 actor.stop()
        #             actor.destroy()

        self.vehicle_actors = []
        self.sensor_actors = []
        self.walker_actors = []
        self.walker_ai_actors = []

        # self.world.tick()
        # self.client.reload_world(reset_settings=True)

    def _set_seed(self, seed):
        if seed:
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)  # Current CPU
            torch.cuda.manual_seed(seed)  # Current GPU
            torch.cuda.manual_seed_all(seed)  # All GPU (Optional)

            self.tm.set_random_device_seed(seed)

    def reset(self, selected_scenario=None, selected_weather=None, seed=None):

        self._clear_all_actors()

        # self.client.reload_world(reset_settings = True)

        if selected_scenario is not None:
            self.reset_num = 0
            self.selected_scenario = selected_scenario

            # print("map:", self.scenario_params[self.selected_scenario]["map"])
            self.client.reload_world(reset_settings=True)
            self.world = self.client.load_world(self.scenario_params[self.selected_scenario]["map"])
            # print("reload done")

        if selected_weather is not None:
            self.selected_weather = selected_weather

        if self.reset_num == 0:

            self._set_dummy_variables()

            # self.world = self.client.load_world(
            #     map_name = self.scenario_params[self.selected_scenario]["map"],
            #     reset_settings = False
            # )
            # remove dynamic objects to prevent 'tables' and 'chairs' flying in the sky
            env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Dynamic)
            toggle1 = set([one_env_obj.id for one_env_obj in env_objs])
            env_objs = self.world.get_environment_objects(carla.CityObjectLabel.Poles)  # street lights
            toggle2 = set([one_env_obj.id for one_env_obj in env_objs])
            objects_to_toggle = toggle1 | toggle2
            self.world.enable_environment_objects(objects_to_toggle, False)
            self.map = self.world.get_map()

            # bp
            self._init_blueprints()

            # spectator
            if self.is_spectator:
                self.spectator = self.world.get_spectator()
            else:
                self.spectator = None

            # tm
            self.tm = self.client.get_trafficmanager(self.carla_tm_port)
            self.tm_port = self.tm.get_port()
            self.tm.set_global_distance_to_leading_vehicle(2.0)
            #
            self._set_seed(seed)
            # lm
            self.lm = self.world.get_lightmanager()
            # self.lm.turn_off(self.lm.get_all_lights())



        # reset
        self.reset_sync_mode(False)
        # self.reset_sync_mode(True)

        self.reset_surrounding_vehicles()
        self.reset_special_vehicles()
        self.reset_walkers()
        self.reset_ego_vehicle()
        self.reset_weather()
        self.reset_sensors()

        self.reset_sync_mode(True)

        # spectator
        if self.spectator is not None:
            # First-perception
            self.spectator.set_transform(
                carla.Transform(self.vehicle.get_transform().location,
                                carla.Rotation(pitch=-float(10), yaw=-float(self.fov)))
            )
            # BEV
            # self.spectator.set_transform(
            #     carla.Transform(self.vehicle.get_transform().location + carla.Location(z=40),
            #                     carla.Rotation(pitch=-90)))

        self.time_step = 0
        self.dist_s = 0
        self.return_ = 0
        self.velocities = []

        self.reward = [0]
        self.perception_data = []
        self.last_action = None

        # MUST warm up !!!!!!
        # take some steps to get ready for the dvs+vidar camera, walkers, and vehicles
        obs = None
        # warm_up_max_steps = self.control_hz     # 15
        warm_up_max_steps = 5
        while warm_up_max_steps > 0:
            warm_up_max_steps -= 1
            obs, _, _, _ = self.step(None)

            # self.world.tick()


            # print("len:self.perception_data:", len(self.perception_data))
            
        # self.vehicle.set_autopilot(True, self.carla_tm_port)
        # while abs(self.vehicle.get_velocity().x) < 0.02:
        #     #             print("!!!take one init step", warm_up_max_steps, self.vehicle.get_control(), self.vehicle.get_velocity())
        #     self.world.tick()
        #     #             action = self.compute_steer_action()
        #     #             obs, _, _, _ = self.step(action=action)
        #     #             self.time_step -= 1
        #     warm_up_max_steps -= 1
        #     if warm_up_max_steps < 0 and self.dvs_data['events'] is not None:
        #         break
        # self.vehicle.set_autopilot(False, self.carla_tm_port)

        self.time_step = 0
        self.init_frame = self.frame
        self.reset_num += 1
        # print("carla env reset done.")

        return obs

    def reset_sync_mode(self, synchronous_mode=True):

        self.delta_seconds = 1.0 / self.max_fps
        # max_substep_delta_time = 0.005

        #         self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=synchronous_mode,
            fixed_delta_seconds=self.delta_seconds,
            # substepping=True, # 'SUBSTEP' mode is not necessary for NONE-DYNAMICS simulations
            # max_substep_delta_time=0.005,
            # max_substeps=int(self.delta_seconds/max_substep_delta_time)
            ))
        self.tm.set_synchronous_mode(synchronous_mode)

    def reset_surrounding_vehicles(self):
        total_surrounding_veh_num = 0
        veh_bp = self.bp_lib.filter('vehicle.*')
        veh_bp = [x for x in veh_bp if int(x.get_attribute('number_of_wheels')) == 4]

        for one_type in ["same_dir_veh", "oppo_dir_veh"]:
            # for one_type in ["same_dir_veh"]:

            one_type_params = self.scenario_params[self.selected_scenario][one_type]

            # print("now at:", one_type)

            for one_part in range(len(one_type_params)):

                veh_num = one_type_params[one_part]["num"]

                while veh_num > 0:
                    if one_type_params[one_part]["type"] != 0:
                        veh_bp = random.choice(one_type_params[one_part]["type"])
                        rand_veh_bp = self.bp_lib.find(veh_bp)
                    else:
                        rand_veh_bp = random.choice(veh_bp)

                    spawn_road_id = one_type_params[one_part]["road_id"]
                    spawn_lane_id = random.choice(
                        one_type_params[one_part]["lane_id"])
                    spawn_start_s = np.random.uniform(
                        one_type_params[one_part]["start_pos"][0],
                        one_type_params[one_part]["start_pos"][1],
                    )

                    veh_pos = self.map.get_waypoint_xodr(
                        road_id=spawn_road_id,
                        lane_id=spawn_lane_id,
                        s=spawn_start_s,
                    ).transform
                    veh_pos.location.z += 0.1

                    if rand_veh_bp.has_attribute('color'):
                        color = random.choice(rand_veh_bp.get_attribute('color').recommended_values)
                        rand_veh_bp.set_attribute('color', color)
                    if rand_veh_bp.has_attribute('driver_id'):
                        driver_id = random.choice(rand_veh_bp.get_attribute('driver_id').recommended_values)
                        rand_veh_bp.set_attribute('driver_id', driver_id)
                    rand_veh_bp.set_attribute('role_name', 'autopilot')
                    vehicle = self.world.try_spawn_actor(rand_veh_bp, veh_pos)

                    if vehicle is not None:
                        vehicle.set_autopilot(True, self.tm_port)
                        if np.random.uniform(0, 1) <= one_type_params[one_part]["beam_ratio"]:
                            vehicle.set_light_state(
                                # carla.VehicleLightState.HighBeam
                                carla.VehicleLightState.All
                            )

                        self.tm.auto_lane_change(vehicle, True)
                        self.tm.vehicle_percentage_speed_difference(
                            vehicle, np.random.uniform(one_type_params[one_part]["speed"][1],
                                                       one_type_params[one_part]["speed"][0]))
                        self.tm.ignore_lights_percentage(vehicle, 100)
                        self.tm.ignore_signs_percentage(vehicle, 100)
                        self.world.tick()

                        self.vehicle_actors.append(vehicle)

                        veh_num -= 1
                        total_surrounding_veh_num += 1
                        # print(f"\t spawn vehicle: {total_surrounding_veh_num}, at {veh_pos.location}")


    def reset_special_vehicles(self):
        special_veh_params = self.scenario_params[self.selected_scenario]["special_veh"]
        veh_bp = self.bp_lib.filter('vehicle.*')
        veh_bp = [x for x in veh_bp if int(x.get_attribute('number_of_wheels')) == 4]

        self.special_veh_lane_ids = []
        for one_part in range(len(special_veh_params)):
            veh_num = special_veh_params[one_part]["num"]

            while veh_num > 0:

                if special_veh_params[one_part]["type"] != 0:
                    # print("@@@:", special_veh_params[one_part]["type"])
                    rand_veh_bp = self.bp_lib.find(special_veh_params[one_part]["type"])
                else:
                    rand_veh_bp = random.choice(veh_bp)

                spawn_road_id = special_veh_params[one_part]["road_id"]
                spawn_lane_id = random.choice(
                    special_veh_params[one_part]["lane_id"])
                spawn_start_s = np.random.uniform(
                    special_veh_params[one_part]["start_pos"][0],
                    special_veh_params[one_part]["start_pos"][1],
                )

                veh_pos = self.map.get_waypoint_xodr(
                    road_id=spawn_road_id,
                    lane_id=spawn_lane_id,
                    s=spawn_start_s,
                ).transform
                veh_pos.location.z += 5
                veh_pos.rotation.pitch = special_veh_params[one_part]["pitch_range"] \
                    if special_veh_params[one_part]["pitch_range"] == 0 \
                    else np.random.uniform(special_veh_params[one_part]["pitch_range"][0],
                                           special_veh_params[one_part]["pitch_range"][1])
                veh_pos.rotation.yaw = special_veh_params[one_part]["yaw_range"] \
                    if special_veh_params[one_part]["yaw_range"] == 0 \
                    else np.random.uniform(special_veh_params[one_part]["yaw_range"][0],
                                           special_veh_params[one_part]["yaw_range"][1])
                veh_pos.rotation.roll = special_veh_params[one_part]["roll_range"] \
                    if special_veh_params[one_part]["roll_range"] == 0 \
                    else np.random.uniform(special_veh_params[one_part]["roll_range"][0],
                                           special_veh_params[one_part]["roll_range"][1])


                if rand_veh_bp.has_attribute('color'):
                    # print("color:", rand_veh_bp.get_attribute('color').recommended_values)
                    if special_veh_params[one_part]["color"] in rand_veh_bp.get_attribute('color').recommended_values:
                        rand_veh_bp.set_attribute('color', special_veh_params[one_part]["color"])
                    else:
                        color = random.choice(rand_veh_bp.get_attribute('color').recommended_values)
                        rand_veh_bp.set_attribute('color', color)

                if rand_veh_bp.has_attribute('driver_id'):
                    driver_id = random.choice(rand_veh_bp.get_attribute('driver_id').recommended_values)
                    rand_veh_bp.set_attribute('driver_id', driver_id)
                rand_veh_bp.set_attribute('role_name', 'autopilot')
                vehicle = self.world.try_spawn_actor(rand_veh_bp, veh_pos)

                if vehicle is not None:
                    self.special_veh_lane_ids.append(spawn_lane_id)

                    vehicle.open_door(carla.VehicleDoor.All)
                    vehicle.set_autopilot(False, self.tm_port)
                    vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                    if np.random.uniform(0, 1) <= special_veh_params[one_part]["beam_ratio"]:
                        vehicle.set_light_state(
                            carla.VehicleLightState.HighBeam
                        )
                    self.world.tick()
                    self.vehicle_actors.append(vehicle)

                    veh_num -= 1



    def reset_walkers(self):
        walker_bp = self.bp_lib.filter('walker.*')
        total_surrounding_walker_num = 0

        walker_params = self.scenario_params[self.selected_scenario]["walker"]

        if len(walker_params) == 0:
            return

        walker_behavior_params = self.scenario_params[self.selected_scenario]["walker_behavior"]

        self.left = carla.WalkerControl(
            direction=carla.Vector3D(y=-1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))
        self.right = carla.WalkerControl(
            direction=carla.Vector3D(y=1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))

        self.forward = carla.WalkerControl(
            direction=carla.Vector3D(x=1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))
        self.backward = carla.WalkerControl(
            direction=carla.Vector3D(x=-1.),
            speed=np.random.uniform(walker_behavior_params["speed"][0], walker_behavior_params["speed"][1]))

        for one_part in range(len(walker_params)):

            walker_num = walker_params[one_part]["num"]

            while walker_num > 0:
                rand_walker_bp = random.choice(walker_bp)

                spawn_road_id = walker_params[one_part]["road_id"]
                spawn_lane_id = random.choice(
                    walker_params[one_part]["lane_id"])
                spawn_start_s = np.random.uniform(
                    walker_params[one_part]["start_pos"][0],
                    walker_params[one_part]["start_pos"][1],
                )

                walker_pos = self.map.get_waypoint_xodr(
                    road_id=spawn_road_id,
                    lane_id=spawn_lane_id,
                    s=spawn_start_s,
                ).transform
                walker_pos.location.z += 0.1

                # set as not invencible
                if rand_walker_bp.has_attribute('is_invincible'):
                    rand_walker_bp.set_attribute('is_invincible', 'false')

                walker_actor = self.world.try_spawn_actor(rand_walker_bp, walker_pos)

                if walker_actor:
                    # walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
                    # walker_controller_actor = self.world.spawn_actor(
                    #     walker_controller_bp, carla.Transform(), walker_actor)
                    # # start walker
                    # walker_controller_actor.start()
                    # # set walk to random point
                    # #             walker_controller_actor.go_to_location(world.get_random_location_from_navigation())
                    # rand_destination = carla.Location(
                    #     x=np.random.uniform(walker_params[one_part]["dest"]["x"][0], walker_params[one_part]["dest"]["x"][1]),
                    #     y=random.choice([walker_params[one_part]["dest"]["y"][0], walker_params[one_part]["dest"]["y"][1]]),
                    #     z=0.
                    # )
                    # walker_controller_actor.go_to_location(rand_destination)
                    # # random max speed (default is 1.4 m/s)
                    # walker_controller_actor.set_max_speed(
                    #     np.random.uniform(
                    #         walker_params[one_part]["speed"][0],
                    #         walker_params[one_part]["speed"][1]
                    #     ))
                    # self.walker_ai_actors.append(walker_controller_actor)

                    self.walker_actors.append(walker_actor)

                    self.world.tick()
                    walker_num -= 1
                    total_surrounding_walker_num += 1
                    # print(f"\t spawn walker: {total_surrounding_walker_num}, at {walker_pos.location}")

    def reset_ego_vehicle(self):

        self.vehicle = None

        # create vehicle
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        #         print(f"\tlen of self.vehicle_polygons: {len(self.vehicle_polygons[-1].keys())}")
        #         print(self.vehicle_polygons[-1].keys())
        ego_veh_params = self.scenario_params[self.selected_scenario]["ego_veh"]

        ego_spawn_times = 0
        max_ego_spawn_times = 10

        while True:
            # print("ego_spawn_times:", ego_spawn_times)d

            if ego_spawn_times > max_ego_spawn_times:

                ego_spawn_times = 0

                # print("\tspawn ego vehicle times > max_ego_spawn_times")
                self._clear_all_actors()
                self.reset_surrounding_vehicles()
                self.reset_special_vehicles()
                self.reset_walkers()

                self.vehicle_polygons = []
                vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
                self.vehicle_polygons.append(vehicle_poly_dict)

                continue

            # Check if ego position overlaps with surrounding vehicles
            overlap = False

            spawn_road_id = ego_veh_params["road_id"]
            if len(self.special_veh_lane_ids) == 0:
                spawn_lane_id = random.choice(ego_veh_params["lane_id"])
            else:
                spawn_lane_id = random.choice(self.special_veh_lane_ids)

            spawn_start_s = np.random.uniform(
                ego_veh_params["start_pos"][0],
                ego_veh_params["start_pos"][1],
            )

            veh_start_pose = self.map.get_waypoint_xodr(
                road_id=spawn_road_id,
                lane_id=spawn_lane_id,
                s=spawn_start_s,
            ).transform
            veh_start_pose.location.z += 0.1


            for idx, poly in self.vehicle_polygons[-1].items():
                poly_center = np.mean(poly, axis=0)
                ego_center = np.array([veh_start_pose.location.x, veh_start_pose.location.y])
                dis = np.linalg.norm(poly_center - ego_center)
                if dis > 8:
                    continue
                else:
                    overlap = True

                    break

            if not overlap:
                self.vehicle = self.world.try_spawn_actor(
                    self.bp_lib.find(ego_veh_params["type"]),
                    veh_start_pose
                )

            if self.vehicle is not None:

                self.vehicle_actors.append(self.vehicle)
                if self.selected_weather == "dense_fog":
                    self.vehicle.set_light_state(carla.VehicleLightState.Fog)
                    self.vehicle.set_light_state(carla.VehicleLightState.HighBeam)
                elif self.selected_scenario == "midnight":
                    self.vehicle.set_light_state(carla.VehicleLightState.HighBeam)

                # AUTO pilot
                if self.ego_auto_pilot:
                    self.vehicle.set_autopilot(True, self.tm_port)

                    self.tm.distance_to_leading_vehicle(self.vehicle, 1)
                    self.tm.auto_lane_change(self.vehicle, True)
                    self.tm.vehicle_percentage_speed_difference(
                        self.vehicle, ego_veh_params["speed"])
                    self.tm.ignore_lights_percentage(self.vehicle, 100)
                    self.tm.ignore_signs_percentage(self.vehicle, 100)
                    # self.tm.force_lane_change(self.vehicle, True)
                else:
                    # immediate running
                    """
                    the driver will spend starting the car engine or changing a new gear. 
                    https://github.com/carla-simulator/carla/issues/3256
                    https://github.com/carla-simulator/carla/issues/1640
                    """
                    # self.vehicle.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
                    # self.world.tick()
                    # self.vehicle.apply_control(carla.VehicleControl(manual_gear_shift=False))

                    physics_control = self.vehicle.get_physics_control()
                    physics_control.gear_switch_time = 0.01
                    physics_control.damping_rate_zero_throttle_clutch_engaged = physics_control.damping_rate_zero_throttle_clutch_disengaged
                    self.vehicle.apply_physics_control(physics_control)
                    self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1, manual_gear_shift=True, gear=1))
                    # pass

                break

            else:
                ego_spawn_times += 1
                time.sleep(0.01)
                # print("ego_spawn_times:", ego_spawn_times)

        self.world.tick()

    def reset_sensors(self):

        # [one_sensor.stop() for one_sensor in self.sensors]

        # data
        if self.BEV:
            self.bev_data = {'frame': 0, 'timestamp': 0.0, 'img': np.zeros((2048, 2048, 3), dtype=np.uint8)}

        if self.TPV:
            self.video_data = {'frame': 0, 'timestamp': 0.0, 'img': np.zeros((1024, 1024, 3), dtype=np.uint8)}

        self.rgb_data = {'frame': 0, 'timestamp': 0.0,
                         'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8)}

        self.depth_data = {'frame': 0, 'timestamp': 0.0,
                           'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 1), dtype=np.uint8)}

        self.dvs_data = {'frame': 0, 'timestamp': 0.0,
                         'events': None,
                         'latest_time': np.zeros(
                             (self.rl_image_size, self.rl_image_size * self.num_cameras), dtype=np.int64),
                         'img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8),
                         'denoised_img': None,
                         # 'rec-img': np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 1), dtype=np.uint8),
                         }


        self.frame = None

        #         def on_tick_func(data):
        #             self.dvs_data["events"] = None
        #         self.world.on_tick(on_tick_func)
        # Bird Eye View
        if self.BEV:
            def __get_bev_data__(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.bev_data['frame'] = data.frame
                self.bev_data['timestamp'] = data.timestamp
                self.bev_data['img'] = array

            self.bev_camera_rgb = self.world.spawn_actor(
                self.bev_camera_bp,
                carla.Transform(carla.Location(z=22), carla.Rotation(pitch=-90, yaw=90, roll=-90)),
                attach_to=self.vehicle)
            self.bev_camera_rgb.listen(lambda data: __get_bev_data__(data))
            self.sensor_actors.append(self.bev_camera_rgb)


        # Third Person View
        if self.TPV:
            def __get_video_data__(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.video_data['frame'] = data.frame
                self.video_data['timestamp'] = data.timestamp
                self.video_data['img'] = array

            self.video_camera_rgb = self.world.spawn_actor(
                self.video_camera_bp,
                carla.Transform(carla.Location(x=-5.5, z=3.5), carla.Rotation(pitch=-15)),
                attach_to=self.vehicle)
            self.video_camera_rgb.listen(lambda data: __get_video_data__(data))
            self.sensor_actors.append(self.video_camera_rgb)

        #         print("\t video sensor init done.")

        # we'll use up to five cameras, which we'll stitch together
        location = carla.Location(x=1, z=1.5)

        # Perception RGB sensor
        def __get_rgb_data__(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.rgb_data['frame'] = data.frame
            self.rgb_data['timestamp'] = data.timestamp
            self.rgb_data['img'] = array

        self.rgb_camera = self.world.spawn_actor(
            self.rgb_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
            attach_to=self.vehicle)
        self.rgb_camera.listen(lambda data: __get_rgb_data__(data))
        self.sensor_actors.append(self.rgb_camera)


        # def __get_depth_data__(data):
        #     # data.convert(carla.ColorConverter.Depth)
        #     data.convert(carla.ColorConverter.LogarithmicDepth) # leading to better precision for small distances at the expense of losing it when further away.
        #     array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
        #     array = np.reshape(array, (data.height, data.width, 4))
        #     array = array[:, :, :3]
        #     array = array[:, :, ::-1]
        #     self.depth_data['frame'] = data.frame
        #     self.depth_data['timestamp'] = data.timestamp
        #     # self.depth_data['img'] = array[:,:,0][..., np.newaxis]
        #     # self.depth_data['img'] = array[:,:,1][..., np.newaxis]
        #     # self.depth_data['img'] = array[:,:,2][..., np.newaxis]
        #     # self.depth_data['img'] = array
        #     self.depth_data['img'] = array[:, :, 0][..., np.newaxis]
        #
        # self.depth_camera = self.world.spawn_actor(
        #     self.depth_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
        #     attach_to=self.vehicle)
        # self.depth_camera.listen(lambda data: __get_depth_data__(data))
        # self.sensor_actors.append(self.depth_camera)

        # Perception DVS sensor
        if self.perception_type.__contains__("DVS"):
            def __get_dvs_data__(data):
                #             print("get_dvs_data:", one_camera_idx)
                events = np.frombuffer(data.raw_data, dtype=np.dtype([
                    ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool_)]))
                self.dvs_data['frame'] = data.frame
                self.dvs_data['timestamp'] = data.timestamp

                # img = np.zeros((data.height, data.width, 3), dtype=np.uint8)
                # img[events[:]['y'], events[:]['x'], events[:]['pol'] * 2] = 255
                # self.dvs_data['img'][:, one_camera_idx * self.rl_image_size: (one_camera_idx + 1) * self.rl_image_size, :] = img

                x = events['x'].astype(np.int32)
                y = events['y'].astype(np.int32)
                p = events['pol'].astype(np.float32)
                t = events['t'].astype(np.float32)
                events = np.column_stack((x, y, p, t))  # (event_num, 4)

                self.dvs_data['events'] = events
                self.dvs_data['events'] = self.dvs_data['events'][np.argsort(self.dvs_data['events'][:, -1])]
                self.dvs_data['events'] = self.dvs_data['events'].astype(np.float32)
                # init done.
                # print(self.dvs_data['events'][:, -1])      # event是按时间递增排的

                # DVS-frame
                img = np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8)
                # print("unique:", np.unique(self.dvs_data['events'][:, 2]))
                img[self.dvs_data['events'][:, 1].astype(np.int),
                    self.dvs_data['events'][:, 0].astype(np.int),
                    self.dvs_data['events'][:, 2].astype(np.int) * 2] = 255
                self.dvs_data['img'] = img
                if self.DENOISE:
                    self.dvs_data['denoised_img'] = img.copy()


            self.dvs_camera = self.world.spawn_actor(
                self.dvs_camera_bp, carla.Transform(location, carla.Rotation(yaw=0.0)),
                attach_to=self.vehicle)
            self.dvs_camera.listen(lambda data: __get_dvs_data__(data))
            self.sensor_actors.append(self.dvs_camera)



        # Collision Sensor
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        self._collision_intensities_during_last_time_step = []

        self.sensor_actors.append(self.collision_sensor)
        #         print("set collision done")

        # print("\t collision sensor init done.")
        self.world.tick()


    def reset_weather(self):
        # # print(self.weather_params.keys())
        # # print(self.selected_weather)
        # assert self.selected_weather in self.weather_params.keys()

        # weather_params = self.weather_params[self.selected_weather]

        # self.weather = self.world.get_weather()

        # self.weather.cloudiness = weather_params["cloudiness"]
        # self.weather.precipitation = weather_params["precipitation"]
        # self.weather.precipitation_deposits = weather_params["precipitation_deposits"]
        # self.weather.wind_intensity = weather_params["wind_intensity"]
        # self.weather.fog_density = weather_params["fog_density"]
        # self.weather.fog_distance = weather_params["fog_distance"]
        # self.weather.wetness = weather_params["wetness"]
        # self.weather.sun_azimuth_angle = weather_params["sun_azimuth_angle"]
        # self.weather.sun_altitude_angle = weather_params["sun_altitude_angle"]

        # self.world.set_weather(self.weather)
        tq=self.selected_weather

        if tq=='clearnoon':
            self.world.set_weather(carla.WeatherParameters.ClearNoon)
        elif tq=='wetsunset':
            self.world.set_weather(carla.WeatherParameters.WetSunset)##
        elif tq=='wetcloudynoon':
            self.world.set_weather(carla.WeatherParameters.WetCloudyNoon)
        elif tq=='softrainsunset':    
            self.world.set_weather(carla.WeatherParameters.SoftRainSunset)
        elif tq=='midrainsunset':
            self.world.set_weather(carla.WeatherParameters.MidRainSunset)
        elif tq=='hardrainnoon':
            self.world.set_weather(carla.WeatherParameters.HardRainNoon)
    def step(self, action):
        rewards = []
        next_obs, done, info = None, None, None

        for _ in range(self.frame_skip):  # default 1
            next_obs, reward, done, info = self._simulator_step(action, self.delta_seconds)
            # next_obs, reward, done, info = self._simulator_step(action)
            rewards.append(reward)
            if done:
                break
        return next_obs, np.mean(rewards), done, info  # just last info?


    def get_reward(self):
        reward = sum(self.reward)
        self.reward = []
        return reward

    def get_perception(self):

        perception_data = self.perception_data

        # 根据perception_type进行预处理
        if self.perception_type.__contains__("rgb"):
            perception_data = np.transpose(perception_data, (2, 0, 1))
            perception_data = (perception_data / 255.).astype(np.float32)
            # print("perception_data:", perception_data.min(), perception_data.max())

        elif self.perception_type.__contains__("dvs"):
            # print("get self.perception_data:", len(self.perception_data))

            # 把events汇总
            events_window = np.concatenate(perception_data, axis=0)
            events_window = events_window[np.argsort(events_window[:, -1])]
            # print("events_window:", events_window.shape)
            # print(events_window)

            perception_data = events_window

            if self.perception_type.__contains__("stream"):
                pass
            elif self.perception_type.__contains__("rec"):
                # print("in dvs rec")

                from run_dvs_rec import run_dvs_rec
                perception_data = run_dvs_rec(
                    # x, y, p, t -> t, x, y, p
                    perception_data[:, [3, 0, 1, 2]],
                    self.rec_model, self.device, self.dvs_rec_args)
                # print(out.shape, out.dtype)
                # print(out[:5, :5])
                # print("raw dvs.shape:", perception_data.shape)
                perception_data = np.transpose(perception_data[..., np.newaxis], (2, 0, 1))
                perception_data = (perception_data / 255.).astype(np.float32)
                # print("perception_data:", perception_data.min(), perception_data.max())

                # print("after dvs rec.shape:", perception_data.shape)

        elif self.perception_type.__contains__("vidar"):
            if self.perception_type.__contains__("stream"):
                pass
            elif self.perception_type.__contains__("rec"):
                # print("in vidar rec")

                from run_vidar_rec import run_vidar_rec
                perception_data = run_vidar_rec(perception_data)
                # print("r", perception_data.shape, perception_data.dtype)
                # print(perception_data)
                perception_data = np.transpose(perception_data[..., np.newaxis], (2, 0, 1))
                # print("vidar rec:", perception_data.min(), perception_data.max())
                # print(perception_data[np.where((perception_data>0) & (perception_data<1))])
                # print("a", perception_data.shape, perception_data.dtype)

        # 重置
        # if self.perception_type.__contains__("rgb"):
        #     pass
        # else:
        #     self.perception_data.clear()
        self.perception_data = []

        return perception_data

    def _control_spectator(self):
        if self.spectator is not None:
            # First-perception
            self.spectator.set_transform(
                carla.Transform(self.vehicle.get_transform().location + carla.Location(z=2),
                                self.vehicle.get_transform().rotation)
            )
            # BEV
            # self.spectator.set_transform(
            #     carla.Transform(self.vehicle.get_transform().location + carla.Location(z=40),
            #                     carla.Rotation(pitch=-90)))


    def _simulator_step(self, action, dt=0.1):

        if action is None and self.last_action is not None:
            action = self.last_action

        if action is not None:
            steer = float(action[0])
            throttle_brake = float(action[1])
            if throttle_brake >= 0.0:
                throttle = throttle_brake
                brake = 0.0
            else:
                throttle = 0.0
                brake = -throttle_brake

            self.last_action = action

        else:
            throttle, steer, brake = 0., 0., 0.

        assert 0.0 <= throttle <= 1.0
        assert -1.0 <= steer <= 1.0
        assert 0.0 <= brake <= 1.0
        vehicle_control = carla.VehicleControl(
            throttle=throttle,  # [0.0, 1.0]
            steer=steer,  # [-1.0, 1.0]
            brake=brake,  # [0.0, 1.0]
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(vehicle_control)

        self._control_spectator()
        self._control_all_walkers()

        # Advance the simulation and wait for the data.
        #         self.dvs_data["events"] = None
        self.frame = self.world.tick()


        # if self.spectator is not None:
        #     self.spectator.set_transform(
        #         carla.Transform(self.ego_vehicle.get_transform().location + carla.Location(z=40),
        #         carla.Rotation(pitch=-90)))

        info = {}
        info['reason_episode_ended'] = ''
        dist_from_center, vel_s, speed, done = self._dist_from_center_lane(self.vehicle, info)
        collision_intensities_during_last_time_step = sum(self._collision_intensities_during_last_time_step)
        self._collision_intensities_during_last_time_step.clear()  # clear it ready for next time step
        assert collision_intensities_during_last_time_step >= 0.
        colliding = float(collision_intensities_during_last_time_step > 0.)

        if colliding:
            self.collide_count += 1
        else:
            self.collide_count = 0

        if self.collide_count >= 20:
            # print("Episode fail: too many collisions ({})! (collide_count: {})".format(speed, self.collide_count))
            info['reason_episode_ended'] = 'collisions'
            done = True

        # reward = vel_s * dt / (1. + dist_from_center) - 1.0 * colliding - 0.1 * brake - 0.1 * abs(steer)
        # collision_cost = 0.0001 * collision_intensities_during_last_time_step

        # [Reward 1]
        # reward = vel_s * dt - collision_cost - abs(steer)

        # [Reward 2]
        # reward = vel_s * dt / (1. + dist_from_center) - 1.0 * colliding - 0.1 * brake - 0.1 * abs(steer)

        # [Reward 3]
        # reward = vel_s * dt / (1. + dist_from_center) - collision_cost - 0.1 * brake - 0.1 * abs(steer)

        # [Reward 4]
        collision_cost = 0.001 * collision_intensities_during_last_time_step
        reward = vel_s * dt - collision_cost - 0.1 * brake - 0.1 * abs(steer)

        self.reward.append(reward)
        # print("vel_s:", vel_s, "speed:", speed)

        self.dist_s += vel_s * self.delta_seconds
        self.return_ += reward

        self.time_step += 1

        next_obs = {
            'RGB-frame': self.rgb_data['img'],
            # 'Depth-frame': self.depth_data['img'],
        }

        if self.TPV:
            next_obs.update({'video-frame': self.video_data['img']})

        if self.BEV:
            next_obs.update({'BEV-frame': self.bev_data['img']})

        # DVS denoising ↓↓↓
        # print(self.dvs_data['events'].shape)
        if self.DENOISE and self.dvs_data['events'].shape[0] >= 500:
            # method = "spatial-temporal correlation filter"
            denoised_img = np.zeros((self.rl_image_size, self.rl_image_size * self.num_cameras, 3), dtype=np.uint8)

            events = self.dvs_data['events']
            X = events[:, 0]
            Y = events[:, 1]
            P = events[:, 2]

            # T = (events[:, 3] - events[0, 3])#  /  (10 ** 6)     # 原本单位是微妙，转为秒
            T = (events[:, 3] - events[0, 3])  /  (10 ** 3)   # 原本单位是微妙，转为毫秒，下面计算不会出现精度问题
            # T = (events[:, 3] - events[0, 3])
            # print("T:", T)
            # print("unique:", np.unique(self.dvs_data['events'][:, 2]))

            method = "motion consistency filter"

            if method == "motion consistency filter":
                # ref: Temporal Up-Sampling for Asynchronous Events
                # highly recommended
                main_events = []
                start_time = time.time()
                # event_num = events.shape[0]      # 不分段
                # event_num = 5000                 # 分段
                # event_num = 10000                # 分段
                event_num = 500                    # 分段
                num = math.ceil(events.shape[0] / event_num)
                for k in range(int(num)):
                    events_at_k = events[k * event_num:(k + 1) * event_num]
                    x = X[k * event_num:(k + 1) * event_num]
                    y = Y[k * event_num:(k + 1) * event_num]
                    p = P[k * event_num:(k + 1) * event_num]
                    t = T[k * event_num:(k + 1) * event_num]

                    t_ref = np.max(t)   # e.g. 49999870
                    # print("t_min:", t_min, "t_ref:", t_ref)

                    rangeX, rangeY = self.rl_image_size * self.num_cameras, self.rl_image_size      # 256, 256
                    flow = contra_max(x, y, p, t, t_ref, rangeX, rangeY)
                    # print("flow:", flow)

                    ref, nx, ny, mx, my = dist_main_noise(flow, x, y, p, t, t_ref, rangeX, rangeY)
                    mxy = np.column_stack((mx, my))
                    # print("xy:", X.shape, Y.shape)
                    # print("mxy:", mx.shape, my.shape)
                    # for iii in range(x.shape[0])
                    for one_event in events_at_k:
                        for one_mxy in mxy:
                            # if one_event[0] == one_mxy[0] and one_event[1] == one_mxy[1]:
                            if np.linalg.norm(one_event[:2] - one_mxy) < 5:   #
                                main_events.append(one_event)
                                break

                main_events = np.array(main_events, dtype=np.float32)
                if main_events.shape[0] != 0:
                    main_events[:, 2][
                        np.where(main_events[:, 2] == -1.)
                    ] = 0       # some bug


            elif method == "spatial-temporal correlation filter":
                # ref: O(N)-Space Spatiotemporal Filter for Reducing Noise in Neuromorphic Vision Sensors

                if len(self.dvs_data['events']) >= 2:

                    # denoising_threshold_t = 100  # 58
                    # denoising_threshold_t = 1000  #  60
                    denoising_threshold_t = 10000  # 50
                    # denoising_threshold_t = 100000  #  64
                    # denoising_threshold_t = 1000000  #  61

                    valid_event_idxs = set(range(len(self.dvs_data['events'])))
                    # print("raw:", len(valid_event_idxs))

                    for one_event_idx, one_event in enumerate(self.dvs_data['events']):

                        self.dvs_data['latest_time'][  # 每个像素点上，存储最新事件时间
                            int(one_event[1]), int(one_event[0])
                        ] = one_event[3]

                        # latest_time = np.zeros(shape=(self.rl_image_size + 2,
                        #                               self.rl_image_size*self.num_cameras + 2), dtype=np.float32)
                        # latest_time[1:self.rl_image_size+1, 1:self.rl_image_size*self.num_cameras+1] = self.dvs_data['latest_time']
                        latest_time = self.dvs_data['latest_time']

                        # latest_time = np.pad(latest_time, ((1,1), (1,1)), 'constant', constant_values=0)
                        # print(latest_time.shape)
                        # kernel_size = 5
                        h_start = one_event[1] - 1
                        h_end = one_event[1] + 2

                        w_start = one_event[0] - 1
                        w_end = one_event[0] + 2

                        if one_event[1] - 1 <= 0:
                            h_start = 0
                        if one_event[0] - 1 <= 0:
                            w_start = 0
                        if one_event[1] + 2 >= self.rl_image_size:
                            h_end = one_event[1] + 1
                        if one_event[0] + 2 >= self.rl_image_size * self.num_cameras:
                            w_end = one_event[0] + 1

                        tmp = latest_time[  # 当前进来事件周围的情况
                              int(h_start): int(h_end),
                              int(w_start): int(w_end),
                              ]
                        new_time = np.max(tmp)
                        #
                        # if new_time == 0:   # valid
                        #     pass
                        # else:
                        if abs(one_event[3] - new_time) < denoising_threshold_t:  # valid
                            pass
                        else:
                            valid_event_idxs -= {one_event_idx}
                            # print("delete:", one_event_idx)

                    # print("after:", len(valid_event_idxs))
                    # print("="*20)
                    # filter
                    # print("events:", len(self.dvs_data['events']), "valid_events:", len(valid_event_idxs))
                    # print("is filter?", len(self.dvs_data['events']) > len(valid_event_idxs))
                    # print()

                    # if len(valid_event_idxs) in valid_event_idxs:
                    #     valid_event_idxs.remove(len(valid_event_idxs))
                    if len(valid_event_idxs) != 0:
                        self.dvs_data['events'] = self.dvs_data['events'][np.array(list(valid_event_idxs))]

                    # DVS denoising ↑↑↑

                    # Some bug
                    self.dvs_data['events'][:, 2][
                        np.where(self.dvs_data['events'][:, 2] == -1.)
                    ] = 0
                    self.dvs_data['events'] = self.dvs_data['events'][np.argsort(self.dvs_data['events'][:, -1])]


            if main_events.shape[0] == 0:
                pass
            else:
                denoised_img[main_events[:, 1].astype(np.int),
                    main_events[:, 0].astype(np.int),
                    main_events[:, 2].astype(np.int) * 2] = 255
                self.dvs_data['denoised_img'] = denoised_img.copy()
                self.dvs_data['events'] = main_events.copy()
            print("raw event number:", events.shape, "denoised event number:", main_events.shape, "spend:", time.time()-start_time, "(s)")

        if self.perception_type == "RGB-frame":
            perception = np.transpose(self.rgb_data['img'], (2, 0, 1))
            # next_obs.update({'perception': perception.copy()})

        elif self.perception_type == "DVS-stream":

            next_obs.update({'DVS-stream': self.dvs_data['events']})
            if self.DENOISE:
                next_obs.update({'Denoised-DVS-frame': self.dvs_data['denoised_img']})
            # events = self.dvs_data['events']
            # # (x, y, p, t)
            # if events is not None:
            #
            #     last_stamp = events[-1, -1]
            #     first_stamp = events[0, -1]
            #     deltaT = last_stamp - first_stamp
            #
            #     if deltaT == 0: deltaT = 1.0
            #
            #     # Iterate over events for a window of dt
            #     for idx in range(events.shape[0]):
            #         e_curr = deepcopy(events[idx])
            #
            #         event_stack.append(e_curr)
            #         t_relative = float(t_final - e_curr[0]) / dt
            #         event_stack[idx - start_idx][0] = t_relative
            #
            #         idx += 1
            #
            #     event_batch_np = np.asarray(event_stack, dtype=np.float32)


        elif self.perception_type == "DVS-frame":
            next_obs.update({'DVS-frame': self.dvs_data['img']})
            if self.DENOISE:
                next_obs.update({'Denoised-DVS-frame': self.dvs_data['denoised_img']})

            # perception = self.dvs_data['img'][:, :, [0, 2]]
            # perception = np.transpose(perception, (2, 0, 1))
            # next_obs.update({'perception': perception.copy()})

        elif self.perception_type == "E2VID-frame":
            next_obs.update({'DVS-frame': self.dvs_data['img']})
            if self.DENOISE:
                next_obs.update({'Denoised-DVS-frame': self.dvs_data['denoised_img']})

            from run_dvs_rec import run_dvs_rec

            dvs_rec_frame = run_dvs_rec(
                # x, y, p, t -> t, x, y, p
                self.dvs_data['events'][:, [3, 0, 1, 2]], self.rec_model,
                self.rl_image_size * self.num_cameras, self.rl_image_size,
                self.device, self.dvs_rec_args)
            # print("dvs_rec_frame:", dvs_rec_frame.shape)  # (84, 420)
            # print("dvs_rec_frame.min():", dvs_rec_frame.min())
            # print("dvs_rec_frame.max():", dvs_rec_frame.max())
            dvs_rec_frame = dvs_rec_frame[..., np.newaxis]  # (84, 420, 1)
            # print("dvs_rec_frame.sahep:", dvs_rec_frame.shape)
            next_obs.update({'E2VID-frame': dvs_rec_frame})
            # next_obs.update({'perception': np.transpose(dvs_rec_frame, (2, 0, 1)).copy()})

        elif self.perception_type == "E2VID-latent":
            pass

        elif self.perception_type == "eVAE-latent":
            pass

        elif self.perception_type == "RGB-frame+DVS-frame":
            next_obs.update({'DVS-frame': self.dvs_data['img']})
            if self.DENOISE:
                next_obs.update({'Denoised-DVS-frame': self.dvs_data['denoised_img']})
            # RGB-frame + DVS-frame
            # next_obs.update({'perception': [
            #     np.transpose(self.rgb_data['img'], (2, 0, 1)).copy(),
            #     np.transpose(self.dvs_data['img'][:, :, [0, 2]], (2, 0, 1)).copy()
            # ]})

        elif self.perception_type == "DVS-voxel-grid" or self.perception_type == "RGB-frame+DVS-voxel-grid":
            next_obs.update({'DVS-frame': self.dvs_data['img']})
            next_obs.update({'events': self.dvs_data["events"]})
            if self.DENOISE:
                next_obs.update({'Denoised-DVS-frame': self.dvs_data['denoised_img']})

            # (5, 84, 84*num_cam)
            num_bins, height, width = 5, self.rl_image_size, int(self.rl_image_size * self.num_cameras)
            voxel_grid = np.zeros(shape=(num_bins, height, width), dtype=np.float32).ravel()

            # get events
            events = self.dvs_data["events"]    # (x, y, p, t)

            if events is not None and len(events) > 0:
                """events to pytorch.tensor"""

                events_torch = torch.from_numpy(events)
                # voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32, device=self.device).flatten()
                voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()
                # normalize the event timestamps so that they lie between 0 and num_bins
                last_stamp = events_torch[-1, -1]
                first_stamp = events_torch[0, -1]
                deltaT = last_stamp - first_stamp

                if deltaT == 0: deltaT = 1.0

                events_torch[:, -1] = (num_bins - 1) * (events_torch[:, -1] - first_stamp) / deltaT
                ts = events_torch[:, -1]
                xs = events_torch[:, 0].long()
                ys = events_torch[:, 1].long()
                pols = events_torch[:, 2].float()
                pols[pols == 0] = -1  # polarity should be +1 / -1

                tis = torch.floor(ts)
                tis_long = tis.long()
                dts = ts - tis
                vals_left = pols * (1.0 - dts.float())
                vals_right = pols * dts.float()

                valid_indices = tis < num_bins
                valid_indices &= tis >= 0
                voxel_grid.index_add_(dim=0,
                                      index=xs[valid_indices] + ys[valid_indices]
                                            * width + tis_long[valid_indices] * width * height,
                                      source=vals_left[valid_indices])

                valid_indices = (tis + 1) < num_bins
                valid_indices &= tis >= 0

                voxel_grid.index_add_(dim=0,
                                      index=xs[valid_indices] + ys[valid_indices] * width
                                            + (tis_long[valid_indices] + 1) * width * height,
                                      source=vals_right[valid_indices])

                voxel_grid = voxel_grid.view(num_bins, height, width)
                voxel_grid = voxel_grid.cpu().numpy()

                # print("voxel_grid:", np.isnan(voxel_grid).all(), voxel_grid.max(), voxel_grid.min())

                """ events to np.array
                # normalize the event timestamps so that they lie between 0 and num_bins
                last_stamp, first_stamp = events[-1, -1], events[0, -1]
                deltaT = last_stamp - first_stamp

                if deltaT == 0:
                    deltaT = 1.0

                events[:, -1] = (num_bins - 1) * (events[:, -1] - first_stamp) / deltaT
                ts = events[:, -1]
                xs = events[:, 0].astype(np.int)
                ys = events[:, 1].astype(np.int)
                pols = events[:, 2]
                pols[pols == 0] = -1  # polarity should be +1 / -1

                tis = ts.astype(np.int)
                dts = ts - tis
                vals_left = pols * (1.0 - dts)
                vals_right = pols * dts

                valid_indices = tis < num_bins
                np.add.at(voxel_grid,
                          xs[valid_indices] + ys[valid_indices] * width +
                          tis[valid_indices] * width * height,
                          vals_left[valid_indices])

                valid_indices = (tis + 1) < num_bins
                np.add.at(voxel_grid,
                          xs[valid_indices] + ys[valid_indices] * width +
                          (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

                voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
                """

                # print("voxel_grid:", np.max(voxel_grid), np.min(voxel_grid))
                next_obs.update({'DVS-voxel-grid': np.transpose(voxel_grid, (1, 2, 0))})

            else:
                voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
                next_obs.update({'DVS-voxel-grid': np.transpose(voxel_grid, (1, 2, 0))})


        info['crash_intensity'] = collision_intensities_during_last_time_step
        info['throttle'] = throttle
        info['steer'] = steer
        info['brake'] = brake
        info['distance'] = vel_s * dt

        if self.time_step >= self.max_episode_steps:
            info['reason_episode_ended'] = 'success'
            # print("Episode success: I've reached the episode horizon ({}).".format(self.max_episode_steps))
            done = True
        #         if speed < 0.02 and self.time_step >= 8 * (self.fps) and self.time_step % 8 * (self.fps) == 0:  # a hack, instead of a counter
        if speed < 0.02 and self.time_step >= self.min_stuck_steps and self.time_step % self.min_stuck_steps == 0:  # a hack, instead of a counter
            # print("Episode fail: speed too small ({}), think I'm stuck! (frame {})".format(speed, self.time_step))
            info['reason_episode_ended'] = 'stuck'
            done = True

        return next_obs, reward, done, info


    def finish(self):
        print('destroying actors.')
        actor_list = self.world.get_actors()
        for one_actor in actor_list:
            one_actor.destroy()
        time.sleep(0.5)
        print('done.')

