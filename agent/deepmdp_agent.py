# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from sac_ae import  Actor, Critic, weight_init, LOG_FREQ
from transition_model import make_transition_model
from decoder import make_decoder

import augmentations
from distance import compute_dist
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
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_weight_lambda=0.0,
        transition_model_type='deterministic',
        num_layers=4,
        num_filters=32
    ):
        self.reconstruction = False
        if decoder_type == 'reconstruction':
            decoder_type = 'pixel'
            self.reconstruction =  True
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
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

        # self.transition_model = make_transition_model(
        #     transition_model_type, encoder_feature_dim*2, action_shape
        # ).to(device)

        # self.reward_decoder = nn.Sequential(
        #     nn.Linear(encoder_feature_dim*2 + action_shape[0], 512),
        #     nn.LayerNorm(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 1)).to(device)

        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim*2 , action_shape
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim*2 + action_shape[0], 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)

        self.transition_model3 = make_transition_model(
            transition_model_type, encoder_feature_dim*2, action_shape
        ).to(device)

        self.reward_decoder3 = nn.Sequential(
            nn.Linear(encoder_feature_dim + action_shape[0], 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)



        self.transition_model1 = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
        ).to(device)

        self.reward_decoder1 = nn.Sequential(
            nn.Linear(encoder_feature_dim*2 + action_shape[0], 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)        

        decoder_params = list(self.transition_model.parameters())  + list(self.reward_decoder.parameters()) + list(self.transition_model3.parameters())  + list(self.reward_decoder3.parameters()) + list(self.transition_model1.parameters())  + list(self.reward_decoder1.parameters())

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type == 'pixel':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)

            self.decoder2 = make_decoder(
                'pixel2', obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder2.apply(weight_init)

            decoder_params += list(self.decoder.parameters())
            decoder_params += list(self.decoder2.parameters())

        self.decoder_optimizer = torch.optim.Adam(
            decoder_params,
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

        self.reward_ex_max=0.00001
        self.reward_in_max=0.00001
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()




    def update_critic(self, obs, action, reward, next_obs, not_done, L, step,cpc_kwargs):
        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)



            obses = cpc_kwargs['obses']
            # actions = cpc_kwargs['actions']
            # curr_rewards = cpc_kwargs['curr_rewards']
            # rewards=cpc_kwargs['rewards']
            # next_obses=cpc_kwargs['next_obses']
            # not_dones=cpc_kwargs['not_dones']

            hx,hy,h0 = self.critic.encoder(obses[:,0,:,:,:].squeeze(),fusion=False)
            hx,hy_d,h1 = self.critic.encoder(obs,fusion=False)
            hx,hy,h2 = self.critic.encoder(obses[:,2,:,:,:].squeeze(),fusion=False)
            diff = torch.abs(h2-h1)+torch.abs(h1-h0)


            cos_dist = F.cosine_similarity(F.normalize(hy_d),F.normalize(diff),dim=1).unsqueeze(dim=1)

            cos_dist = 0.2*np.exp(-2e-5*step)*(-cos_dist)
            L.log('train_alpha/cos_dist', cos_dist.mean(), step)

            instrinsic = cos_dist*self.reward_ex_max/self.reward_in_max
            # print('instrinsic',instrinsic)
            # print('tq:',target_Q)
            # print('reward:',reward)

            L.log('train_alpha/instrinsic', instrinsic.mean(), step)

            self.reward_in_max = max(max(instrinsic), self.reward_in_max)
            self.reward_ex_max = max(max(reward), self.reward_ex_max)

            target_Q = target_Q + instrinsic

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)



    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)


        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_transition_reward_model(self, obs,obs_aug,  action, next_obs, reward, L, step):



        h, h3,h1 = self.critic.encoder(obs, fusion=False)
        # import pdb
        # pdb.set_trace()
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model1(torch.cat([h3, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h,next_h3,next_h1 = self.critic.encoder(next_obs, fusion=False)
        diff = (pred_next_latent_mu - next_h3.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        L.log('train_ae/transition_loss', loss, step)

        pred_next_reward = self.reward_decoder(torch.cat([h, action], dim=1))
        reward_loss = F.mse_loss(pred_next_reward, reward)
        L.log('train_ae/reward_loss', reward_loss, step)
        total_loss = loss + reward_loss


        #con
        z_fusion, z_v,z_i = h, h3,h1 #self.critic.encoder(obs, fusion=False)
        # with torch.no_grad():
        #     z_fusion_p, z_v_p,z_i_p = self.critic_target.encoder(obs_aug, fusion=False)
        z_fusion_p, z_v_p,z_i_p = self.critic.encoder(obs_aug, fusion=False)


        curl_loss3 = self.contrastive_loss_forward(z_i,z_i_p)

        L.log('train_ae/curl_loss3', curl_loss3, step)
        # L.log('train_ae/curl_loss4', curl_loss4, step)


        # total_loss = total_loss + 0.01*(curl_loss1+curl_loss3)
        total_loss = total_loss + 0.05*(curl_loss3)






        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()





    def contrastive_loss_forward(self,
                                  hidden1: torch.Tensor,
                                  hidden2: torch.Tensor,
                                  hidden_norm: bool = True,
                                  temperature: float = 1.0):
        """
        hidden1: (batch_size, dim)
        hidden2: (batch_size, dim)
        """
        LARGE_NUM = 1e9
        batch_size, hidden_dim = hidden1.shape
        
        if hidden_norm:
            hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
            hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.arange(0, batch_size).to(device=hidden1.device)
        masks = torch.nn.functional.one_hot(torch.arange(0, batch_size), num_classes=batch_size).to(device=hidden1.device, dtype=torch.float)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature  # shape (batch_size, batch_size)

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b
        return loss



    def update_decoder(self, obs, obs_aug, action, target_obs, L, step):  #  uses transition model
        # image might be stacked, just grab the first 3 (rgb)!
        assert target_obs.dim() == 4
        # target_obs = target_obs[:, :3, :, :]
        # import pdb
        # pdb.set_trace()

        h, h3,h1 = self.critic.encoder(obs, fusion=False)
        h_aug, h3_aug,h1_aug = self.critic.encoder(obs_aug, fusion=False)



        obs_phase =  obs[:,3:6,:,:]

        rec_obs = self.decoder2(h1)
        loss2 = F.mse_loss(obs_phase,  rec_obs)


        rec_obs_aug = self.decoder2(h1_aug)
        loss2_aug = F.mse_loss(obs_phase,  rec_obs_aug)

        loss = loss2 + loss2_aug #+ loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)



    def update(self, replay_buffer, L, step):
        # obs, action, _, reward, next_obs, not_done, dict_obs = replay_buffer.sample_ctmr()
        cpc_kwargs = replay_buffer.sample_ctmr()

        # i=1

        obses = cpc_kwargs['obses']
        actions = cpc_kwargs['actions']
        curr_rewards = cpc_kwargs['curr_rewards']
        rewards=cpc_kwargs['rewards']
        next_obses=cpc_kwargs['next_obses']
        not_dones=cpc_kwargs['not_dones']
        i=1


        obs = obses[:,i,:,:,:]
        action = actions[:,i,:]
        next_obs = next_obses[:,i,:,:,:]
        reward = rewards[:,i,:]
        not_done = not_dones[:,i,:]


        obs_aug = augmentations.random_conv(obs)

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step,cpc_kwargs)
        self.update_transition_reward_model(obs,obs_aug, action, next_obs, reward, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:  # decoder_type is pixel
            # import pdb
            # pdb.set_trace()
            self.update_decoder(obs,obs_aug, action, next_obs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
