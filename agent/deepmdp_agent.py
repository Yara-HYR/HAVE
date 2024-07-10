# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
# import info_nce


from utils.soft_update_params import soft_update_params
from utils.preprocess_obs import preprocess_obs

from sac_ae import Actor, Critic, weight_init, LOG_FREQ
from transition_model import make_transition_model
from decoder import make_decoder
from action_model import make_action_model
from refusion import refusion
from fusion import fusion
import copy
plt.switch_backend('agg')


class DeepMDPAgent(object):
    """Baseline algorithm with transition model and various decoder types."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        perception_type='RGB-frame',
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        action_model_update_freq=1,
        transition_reward_model_update_freq=1,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_weight_lambda=0.0,
        transition_model_type='deterministic',
        num_layers=4,
        num_filters=32,
        LOG_FREQ=5000,
    ):
        # self.fig, [self.ax1, self.ax2] = plt.subplots(1, 2)

        self.similarity = nn.CosineSimilarity(dim=1)

        self.LOG_FREQ = LOG_FREQ
        self.obs_shape = obs_shape
        self.reconstruction = False
        self.encoder_type = encoder_type
        self.action_model_update_freq = action_model_update_freq

        if decoder_type == 'reconstruction':
            decoder_type = 'pixel'
            self.reconstruction = True
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.transition_reward_model_update_freq = transition_reward_model_update_freq
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_type = decoder_type

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())




        if encoder_type == "pixelCat":
            self.transition_model_rgb = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            self.transition_model_dvs = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            # self.transition_model_con = make_transition_model(
            #     transition_model_type, encoder_feature_dim*2, action_shape, encoder_feature_dim*2, contain_action=True
            # ).to(device)
            self.transition_model_rgb2 = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            self.transition_model_dvs2 = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim, contain_action=True
            ).to(device)
            # self.reward_decoder_rgb = nn.Sequential(
            #     nn.Linear(encoder_feature_dim + action_shape[0], 512),
            #     nn.LayerNorm(512),
            #     nn.ReLU(),
            #     nn.Linear(512, 1)).to(device)
            # self.reward_decoder_dvs = nn.Sequential(
            #     nn.Linear(encoder_feature_dim + action_shape[0], 512),
            #     nn.LayerNorm(512),
            #     nn.ReLU(),
            #     nn.Linear(512, 1)).to(device)
            self.reward_decoder_con = nn.Sequential(
                nn.Linear(encoder_feature_dim*2 + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)

            # decoder_params = list(self.transition_model_rgb.parameters()) + \
            #                  list(self.transition_model_dvs.parameters()) + \
            #                  list(self.transition_model_con.parameters()) + \
            #                  list(self.reward_decoder_rgb.parameters()) + \
            #                  list(self.reward_decoder_dvs.parameters()) + \
            #                  list(self.reward_decoder_con.parameters())
            decoder_params = list(self.transition_model_rgb.parameters()) + \
                             list(self.transition_model_dvs.parameters()) + \
                             list(self.transition_model_rgb2.parameters()) + \
                             list(self.transition_model_dvs2.parameters()) + \
                             list(self.reward_decoder_con.parameters())
            self.decoder = None

            if self.reconstruction:

                self.decoder_rgb = make_decoder(
                    decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 15
                ).to(device)
                self.decoder_dvs = make_decoder(
                    # decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 1
                    decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters, 9  # 5*3
                ).to(device)
                self.decoder_rgb.apply(weight_init)
                self.decoder_dvs.apply(weight_init)

                decoder_params += list(self.decoder_rgb.parameters())
                decoder_params += list(self.decoder_dvs.parameters())



        else:

            self.action_model = None
            self.transition_model = make_transition_model(
                transition_model_type, encoder_feature_dim, action_shape, encoder_feature_dim
            ).to(device)

            self.reward_decoder = nn.Sequential(
                nn.Linear(encoder_feature_dim + action_shape[0], 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)
            self.decoder = None

            decoder_params = list(self.transition_model.parameters()) + \
                             list(self.reward_decoder.parameters())

            if perception_type == "RGB-frame":
                outtttt = 9
            elif perception_type == "DVS-voxel-grid":
                outtttt = 15
            else:
                outtttt = 3

            if self.reconstruction:

                self.decoder = make_decoder(
                    decoder_type, obs_shape, encoder_feature_dim,
                    num_layers, num_filters, outtttt
                ).to(device)
                self.decoder.apply(weight_init)
                decoder_params += list(self.decoder.parameters())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder_q)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # self.decoder = None
        # if decoder_type == 'pixel':
        #     # create decoder
        #     if encoder_type == "pixelHybridActionMaskV5":
        #         decoder_type = "HybridActionMaskV5Decoder"
        #
        #     self.decoder = make_decoder(
        #         decoder_type, obs_shape,
        #         encoder_feature_dim if encoder_type != "pixelMultiHeadHybridMask" else encoder_feature_dim * 2,
        #         num_layers,
        #         num_filters
        #     ).to(device)
        #     self.decoder.apply(weight_init)
        #     decoder_params += list(self.decoder.parameters())

        self.decoder_optimizer = torch.optim.Adam(
            decoder_params,
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        print("self.critic.encoder_q.parameters():", self.critic.encoder_q.parameters())
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder_q.parameters(), lr=encoder_lr
        )

        emb_shape=100




        # self.mixer_c = QMixer(emb_shape,device,n_agents=2,mixing_embed_dim=32,hypernet_embed=64)
        # self.mixer_t =  copy.deepcopy(self.mixer_c)#QMixer(emb_shape,device,n_agents=2,mixing_embed_dim=32,hypernet_embed=64)

        self.mixer_c = fusion(emb_shape,device,n_agents=2,mixing_embed_dim=32,hypernet_embed=64,agent_own_state_size=50)
        self.mixer_t = copy.deepcopy(self.mixer_c)

        self.mixer_c2 = refusion(emb_shape,device,n_agents=2,mixing_embed_dim=32,hypernet_embed=64)
        self.mixer_t2 =  copy.deepcopy(self.mixer_c2)#QMixer(emb_shape,device,n_agents=2,mixing_embed_dim=32,hypernet_embed=64)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        critic_params = list(self.critic.parameters())+list(self.mixer_c.parameters())

        self.critic_optimizer = torch.optim.Adam(
            critic_params, lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if hasattr(self, 'decoder_rgb'):
            self.decoder_rgb.train(training)
        if hasattr(self, 'decoder_dvs'):
            self.decoder_dvs.train(training)

        # if self.decoder is not None:
        #     self.decoder.train(training)
        # if len(self.decoder) != 0:
        #     for one_decoder in self.decoder:
        #         one_decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _obs_to_input(self, obs):
        if isinstance(obs, list) and len(obs) == 2:
            rgb_obs = torch.FloatTensor(obs[0]).to(self.device)
            dvs_obs = torch.FloatTensor(obs[1]).to(self.device)
            rgb_obs = rgb_obs.unsqueeze(0)
            dvs_obs = dvs_obs.unsqueeze(0)

            _obs = [rgb_obs, dvs_obs]
        elif isinstance(obs, list) and len(obs) == 3:
            rgb_obs = torch.FloatTensor(obs[0]).to(self.device)
            dvs_obs = torch.FloatTensor(obs[1]).to(self.device)
            depth_obs = torch.FloatTensor(obs[2]).to(self.device)
            rgb_obs = rgb_obs.unsqueeze(0)
            dvs_obs = dvs_obs.unsqueeze(0)
            depth_obs = depth_obs.unsqueeze(0)

            _obs = [rgb_obs, dvs_obs, depth_obs]

        elif isinstance(obs, list) and len(obs) == 4:
            rgb_obs = torch.FloatTensor(obs[0]).to(self.device)
            dvs_obs = torch.FloatTensor(obs[1]).to(self.device)
            depth_obs = torch.FloatTensor(obs[2]).to(self.device)
            dvs_obs2 = torch.FloatTensor(obs[3]).to(self.device)

            rgb_obs = rgb_obs.unsqueeze(0)
            dvs_obs = dvs_obs.unsqueeze(0)
            depth_obs = depth_obs.unsqueeze(0)
            dvs_obs2 = dvs_obs2.unsqueeze(0)

            _obs = [rgb_obs, dvs_obs, depth_obs, dvs_obs2]

        else:
            _obs = torch.FloatTensor(obs).to(self.device)
            _obs = _obs.unsqueeze(0)
        return _obs

    def select_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _, _ = self.actor(
                _obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _,_ = self.actor(next_obs, global_f=True)
            target_Q1, target_Q2, next_global_f = self.critic_target(next_obs, policy_action, global_f=True)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

            # print("policy_action.nan:", torch.any(torch.isnan(policy_action)))
            # print("target_Q1.nan:", torch.any(torch.isnan(target_Q1)))
            # print("target_Q2.nan:", torch.any(torch.isnan(target_Q2)))

        # get current Q estimates
        current_Q1, current_Q2, global_f = self.critic(obs, action, detach_encoder=False, global_f=True)
        # print("current_Q1.nan:", torch.any(torch.isnan(current_Q1)))
        # print("current_Q2.nan:", torch.any(torch.isnan(current_Q2)))

        with torch.no_grad():
            # mu_rgb, policy_action, log_pi, _ = self.actor(next_obs,rgb=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action,rgb=True)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q_rgb = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1_rgb, current_Q2_rgb = self.critic(obs, action, detach_encoder=False,rgb=True)




        with torch.no_grad():
            # mu_dvs, policy_action, log_pi, _ = self.actor(next_obs,dvs=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action,dvs=True)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q_dvs = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1_dvs, current_Q2_dvs = self.critic(obs, action, detach_encoder=False,dvs=True)

        # critic_loss1 = F.mse_loss(current_Q1_rgb, target_Q_rgb) + F.mse_loss(current_Q2_rgb, target_Q_rgb)
        # critic_loss2 = F.mse_loss(current_Q1_dvs, target_Q_dvs) + F.mse_loss(current_Q2_dvs, target_Q_dvs)
        # critic_loss = critic_loss1+critic_loss2

        current_Q1_total = self.mixer_c([current_Q1_rgb, current_Q1_dvs], global_f)
        current_Q2_total = self.mixer_c([current_Q2_rgb, current_Q2_dvs], global_f)
        with torch.no_grad():
            target_Q_total = self.mixer_t([target_Q_rgb, target_Q_dvs], next_global_f)



        current_Q1_f = self.mixer_c2([current_Q1, current_Q1_total],global_f)
        current_Q2_f = self.mixer_c2([current_Q2, current_Q2_total],global_f)

        with torch.no_grad():
            target_Q_f = self.mixer_t2([target_Q, target_Q_total], next_global_f)


        critic_loss5 =  F.mse_loss(current_Q1_f,
                                  target_Q_f) + F.mse_loss(current_Q2_f, target_Q_f)
       
        critic_loss = critic_loss5

        L.log('train_critic/loss5', critic_loss5, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step, log_freq=self.LOG_FREQ)
    # def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
    #     with torch.no_grad():
    #         _, policy_action, log_pi, _ = self.actor(next_obs)
    #         target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
    #         target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
    #         target_Q = reward + (not_done * self.discount * target_V)

    #         # print("policy_action.nan:", torch.any(torch.isnan(policy_action)))
    #         # print("target_Q1.nan:", torch.any(torch.isnan(target_Q1)))
    #         # print("target_Q2.nan:", torch.any(torch.isnan(target_Q2)))

    #     # get current Q estimates
    #     current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
    #     # print("current_Q1.nan:", torch.any(torch.isnan(current_Q1)))
    #     # print("current_Q2.nan:", torch.any(torch.isnan(current_Q2)))

    #     critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
    #     L.log('train_critic/loss', critic_loss, step)

    #     # Optimize the critic
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()

    #     self.critic.log(L, step, log_freq=self.LOG_FREQ)



    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2, global_f = self.critic(obs, pi, detach_encoder=True,global_f=True)
        actor_Q1_rgb, actor_Q2_rgb = self.critic(obs, pi, detach_encoder=True,rgb=True)
        actor_Q1_dvs, actor_Q2_dvs = self.critic(obs, pi, detach_encoder=True,dvs=True)

        actor_Q1_tot = self.mixer_c([actor_Q1_rgb, actor_Q1_dvs], global_f)
        actor_Q2_tot = self.mixer_c([actor_Q2_rgb, actor_Q2_dvs], global_f)

        actor_Q1_f = self.mixer_c2([actor_Q1, actor_Q1_tot], global_f)
        actor_Q2_f = self.mixer_c2([actor_Q2, actor_Q2_tot], global_f)

        actor_Q = torch.min(actor_Q1_f, actor_Q2_f)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_transition_reward_model_pixelCat(self, obs, action, next_obs, reward, L, step):
        con_h, [rgb_h, dvs_h] = self.critic.encoder_q(obs)
        next_con_h, [next_rgb_h, next_dvs_h] = self.critic.encoder_q(next_obs)







        pred_next_rgb_latent_mu, pred_next_rgb_latent_sigma = self.transition_model_rgb2(torch.cat([rgb_h, action], dim=1))
        pred_next_dvs_latent_mu, pred_next_dvs_latent_sigma = self.transition_model_dvs2(torch.cat([dvs_h, action], dim=1))
        # pred_next_con_latent_mu, pred_next_con_latent_sigma = self.transition_model_con(torch.cat([con_h, action], dim=1))
        if pred_next_rgb_latent_sigma is None: pred_next_rgb_latent_sigma = torch.ones_like(pred_next_rgb_latent_mu)
        if pred_next_dvs_latent_sigma is None: pred_next_dvs_latent_sigma = torch.ones_like(pred_next_dvs_latent_mu)
        if pred_next_con_latent_sigma is None: pred_next_con_latent_sigma = torch.ones_like(pred_next_con_latent_mu)


        rgb_diff = (pred_next_rgb_latent_mu - next_rgb_h.detach()) / pred_next_rgb_latent_sigma
        dvs_diff = (pred_next_dvs_latent_mu - next_dvs_h.detach()) / pred_next_dvs_latent_sigma
        con_diff = (pred_next_con_latent_mu - next_con_h.detach()) / pred_next_con_latent_sigma
        rgb_transition_loss = torch.mean(0.5 * rgb_diff.pow(2) + torch.log(pred_next_rgb_latent_sigma))
        dvs_transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_dvs_latent_sigma))
        con_transition_loss = torch.mean(0.5 * con_diff.pow(2) + torch.log(pred_next_con_latent_sigma))
        transition_loss2 = rgb_transition_loss + dvs_transition_loss + con_transition_loss
        L.log('train_ae/transition_loss2', transition_loss2, step)



        pred_next_con_reward = self.reward_decoder_con(torch.cat([con_h, action], dim=1))
        con_reward_loss = F.mse_loss(pred_next_con_reward, reward)
        reward_loss =   con_reward_loss
        L.log('train_ae/reward_loss', reward_loss, step)


        total_loss = reward_loss + transition_loss2

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_transition_reward_model_pixelCatSep(self, obs, action, next_obs, reward, L, step):
        fusion_h, [rgb_h, dvs_h] = self.critic.encoder_q(obs)
        next_fusion_h, [next_rgb_h, next_dvs_h] = self.critic.encoder_q(next_obs)

        pred_next_rgb_latent_mu, pred_next_rgb_latent_sigma = self.transition_model_rgb(torch.cat([rgb_h, action], dim=1))
        pred_next_dvs_latent_mu, pred_next_dvs_latent_sigma = self.transition_model_dvs(torch.cat([dvs_h, action], dim=1))
        if pred_next_rgb_latent_sigma is None: pred_next_rgb_latent_sigma = torch.ones_like(pred_next_rgb_latent_mu)
        if pred_next_dvs_latent_sigma is None: pred_next_dvs_latent_sigma = torch.ones_like(pred_next_dvs_latent_mu)

        rgb_diff = (pred_next_rgb_latent_mu - next_rgb_h.detach()) / pred_next_rgb_latent_sigma
        dvs_diff = (pred_next_dvs_latent_mu - next_dvs_h.detach()) / pred_next_dvs_latent_sigma
        rgb_transition_loss = torch.mean(0.5 * rgb_diff.pow(2) + torch.log(pred_next_rgb_latent_sigma))
        dvs_transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_dvs_latent_sigma))
        transition_loss = rgb_transition_loss + dvs_transition_loss
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_rgb_reward = self.reward_decoder_rgb(torch.cat([rgb_h, action], dim=1))
        pred_next_dvs_reward = self.reward_decoder_dvs(torch.cat([dvs_h, action], dim=1))
        rgb_reward_loss = F.mse_loss(pred_next_rgb_reward, reward)
        dvs_reward_loss = F.mse_loss(pred_next_dvs_reward, reward)
        reward_loss = rgb_reward_loss + dvs_reward_loss
        L.log('train_ae/reward_loss', reward_loss, step)


        total_loss = transition_loss + reward_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    def update_transition_reward_model_pixelHybridActionMask(self, obs, action, next_obs, reward, L, step):
        # fusion_h, [rgb_h, dvs_h, texture_h, action_mask] \
        fusion_h, _ = self.critic.encoder_q(obs)
            # next_fusion_h, [next_rgb_h, next_dvs_h, next_texture_h, next_action_mask] \
        next_fusion_h, _ = self.critic.encoder_q(next_obs)

        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([fusion_h, action], dim=1))
        # pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([rgb_h, dvs_h, action], dim=1))
        if pred_next_latent_sigma is None: pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
        dvs_diff = (pred_next_latent_mu - next_fusion_h.detach()) / pred_next_latent_sigma
        # dvs_diff = (pred_next_latent_mu - torch.cat([next_rgb_h, next_dvs_h], dim=1).detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_cat_reward = self.reward_decoder_cat(torch.cat([fusion_h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_cat_reward, reward)
        L.log('train_ae/reward_loss', reward_loss, step)

        # total_loss = 10 * transition_loss + reward_loss# + ae_loss + action_loss
        total_loss = transition_loss + reward_loss# + ae_loss + action_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    def update_transition_reward_model_pixelHybridActionMaskV2(self, obs, action, next_obs, reward, L, step):
        fusion_h, _ = self.critic.encoder_q(obs)
        next_fusion_h, _ = self.critic.encoder_q(next_obs)

        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([fusion_h, action], dim=1))
        # pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([rgb_h, dvs_h, action], dim=1))
        if pred_next_latent_sigma is None: pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
        dvs_diff = (pred_next_latent_mu - next_fusion_h.detach()) / pred_next_latent_sigma
        # dvs_diff = (pred_next_latent_mu - torch.cat([next_rgb_h, next_dvs_h], dim=1).detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_cat_reward = self.reward_decoder_cat(torch.cat([fusion_h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_cat_reward, reward)
        L.log('train_ae/reward_loss', reward_loss, step)

        # total_loss = 10 * transition_loss + reward_loss# + ae_loss + action_loss
        total_loss = transition_loss + reward_loss# + ae_loss + action_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_transition_reward_model_pixelHybridActionMaskV3(self, obs, action, next_obs, reward, L, step):
        fusion_h, _ = self.critic.encoder_q(obs)
        next_fusion_h, _ = self.critic.encoder_q(next_obs)

        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([fusion_h, action], dim=1))
        # pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([rgb_h, dvs_h, action], dim=1))
        if pred_next_latent_sigma is None: pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
        dvs_diff = (pred_next_latent_mu - next_fusion_h.detach()) / pred_next_latent_sigma
        # dvs_diff = (pred_next_latent_mu - torch.cat([next_rgb_h, next_dvs_h], dim=1).detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_cat_reward = self.reward_decoder_cat(torch.cat([fusion_h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_cat_reward, reward)
        L.log('train_ae/reward_loss', reward_loss, step)

        # total_loss = 10 * transition_loss + reward_loss# + ae_loss + action_loss
        total_loss = transition_loss + reward_loss# + ae_loss + action_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_transition_reward_model_pixelHybridActionMaskV4(self, obs, action, next_obs, reward, L, step):
        fusion_h, _ = self.critic.encoder_q(obs)
        next_fusion_h, _ = self.critic.encoder_q(next_obs)

        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([fusion_h, action], dim=1))
        # pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([rgb_h, dvs_h, action], dim=1))
        if pred_next_latent_sigma is None: pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
        dvs_diff = (pred_next_latent_mu - next_fusion_h.detach()) / pred_next_latent_sigma
        # dvs_diff = (pred_next_latent_mu - torch.cat([next_rgb_h, next_dvs_h], dim=1).detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_cat_reward = self.reward_decoder_cat(torch.cat([fusion_h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_cat_reward, reward)
        L.log('train_ae/reward_loss', reward_loss, step)

        # total_loss = 10 * transition_loss + reward_loss# + ae_loss + action_loss
        total_loss = transition_loss + reward_loss# + ae_loss + action_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    def update_transition_reward_model_pixelHybridActionMaskV5(self, obs, action, next_obs, reward, L, step):
        fusion_h, _ = self.critic.encoder_q(obs)
        next_fusion_h, _ = self.critic.encoder_q(next_obs)
        # print("fusion_h:", fusion_h)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([fusion_h, action], dim=1))
        # pred_next_latent_mu, pred_next_latent_sigma = self.transition_model_cat(torch.cat([rgb_h, dvs_h, action], dim=1))
        if pred_next_latent_sigma is None: pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
        dvs_diff = (pred_next_latent_mu - next_fusion_h.detach()) / pred_next_latent_sigma
        # dvs_diff = (pred_next_latent_mu - torch.cat([next_rgb_h, next_dvs_h], dim=1).detach()) / pred_next_latent_sigma
        transition_loss = torch.mean(0.5 * dvs_diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_cat_reward = self.reward_decoder_cat(torch.cat([fusion_h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_cat_reward, reward)
        L.log('train_ae/reward_loss', reward_loss, step)

        # print("T_Loss:", transition_loss, "R_Loss:", reward_loss)
        # total_loss = 10 * transition_loss + reward_loss# + ae_loss + action_loss
        total_loss = transition_loss + reward_loss# + ae_loss + action_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def update_inconsistency(self, obs, action, target_obs, L, step):
        if self.encoder_type == "pixelWAE":
            ###############################################################
            fusion_h, [rgb_h, dvs_h,
                       rgb_T_A, rgb_T_E, dvs_T_A, dvs_T_E,
                       rgb_R_A, rgb_R_E, dvs_R_A, dvs_R_E,
                       # fuse_T, fuse_R,
                       sum_T, sum_R
                       ] = self.critic.encoder_q(obs)

            # 拉大两个模态不一致部分的距离
            # loss_TE1 = (rgb_T_E * dvs_T_E).mean()
            # loss_RE1 = (rgb_R_E * dvs_R_E).mean()
            # 相同时为1，正交时为0，相反时为-1!!!!!!!!!!!!!!!!!!!
            loss_TE1 = self.similarity(rgb_T_E, dvs_T_E).mean()
            loss_RE1 = self.similarity(rgb_R_E, dvs_R_E).mean()
            # loss_TE1 = - F.mse_loss(rgb_T_E, dvs_T_E)
            # loss_RE1 = - F.mse_loss(rgb_R_E, dvs_R_E)
            # L.log('train_ae/inter-incon', loss_TE1 + loss_RE1, step)


            # intra inconsistency
            # loss_TE2 = (rgb_T_E * rgb_T_E).mean()
            # loss_RE2 = (dvs_T_E * dvs_T_E).mean()
            # L.log('train_ae/intra-incon', loss_TE2+loss_RE2, step)
            #
            aaa, bbb = 1, 0.1
            total_loss = aaa * (loss_TE1 + loss_RE1)# + \
                         # bbb * (loss_TE2 + loss_RE2)

            L.log('train_ae/inconsistency', total_loss, step)

            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            total_loss.backward()
            # total_loss.backward(retain_graph=True)
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.critic.encoder_q.parameters(), self.critic.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.M + param_q.data * (1.0 - self.M)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, rgb_keys, dvs_keys):
        # gather keys before updating queue
        # multiple gpu ↓
        # keys = concat_all_gather(keys)

        batch_size = rgb_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue_rgb[:, ptr : ptr + batch_size] = rgb_keys.T
        self.queue_dvs[:, ptr : ptr + batch_size] = dvs_keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr



    def update_transition_reward_model(self, obs, action, next_obs, reward, L, step):
        h, _ = self.critic.encoder_q(obs)
        # print("h.nan:", torch.any(torch.isnan(h)))

        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h, _ = self.critic.encoder_q(next_obs)

        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma

        transition_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        # print("diff.pow(2):", diff.pow(2))
        # print("transition_loss:", transition_loss)
        L.log('train_ae/transition_loss', transition_loss, step)

        pred_next_reward = self.reward_decoder(torch.cat([h, action], dim=1))
        # print("pred_next_reward:", pred_next_reward)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        # print("reward_loss:", reward_loss)

        L.log('train_ae/reward_loss', reward_loss, step)

        total_loss = transition_loss + reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()




    def update(self, replay_buffer, L, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample(
            multi=True if isinstance(self.obs_shape, list) and len(self.obs_shape) >= 2 else False)

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.transition_reward_model_update_freq == 0:
            if self.encoder_type == "pixelCat":


                self.update_transition_reward_model_pixelCat(obs, action, next_obs, reward, L, step)

            else:
                self.update_transition_reward_model(obs, action, next_obs, reward, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            soft_update_params(
                self.critic.encoder_q, self.critic_target.encoder_q,
                self.encoder_tau
            )
            soft_update_params(
                self.mixer_c, self.mixer_t,
                self.encoder_tau
            )
            soft_update_params(
                self.mixer_c2, self.mixer_t2,
                self.encoder_tau
            )

        
        




    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        # if self.decoder is not None:
        #     torch.save(
        #         self.decoder.state_dict(),
        #         '%s/decoder_%s.pt' % (model_dir, step)
        #     )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        # if self.decoder is not None:
        #     self.decoder.load_state_dict(
        #         torch.load('%s/decoder_%s.pt' % (model_dir, step))
        #     )

