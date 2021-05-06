import os
from PIL import Image
import numpy as np
import torch
from torch import nn

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims, valid_mean
from rlpyt.utils.averages import RunningMeanStd, RewardForwardFilter
from rlpyt.models.utils import Flatten
from rlpyt.models.curiosity.encoders import BurdaHead, MazeHead, UniverseHead
import cv2


class RND(nn.Module):
    """Curiosity model for intrinsically motivated agents: 
    """

    def __init__(
            self, 
            image_shape, 
            obs_stats=None,
            prediction_beta=1.0,
            drop_probability=1.0,
            gamma=0.99,
            device='cpu'
            ):
        super(RND, self).__init__()

        self.prediction_beta = prediction_beta
        self.drop_probability = drop_probability
        self.device = torch.device('cuda:0' if device == 'gpu' else 'cpu')

        c, h, w = 1, image_shape[1], image_shape[2] # assuming grayscale inputs
        self.c = c
        self.h = h
        self.w = w
        self.obs_rms = RunningMeanStd(shape=(1, c, h, w)) # (T, B, c, h, w)
        if obs_stats is not None:
            self.obs_rms.mean[0] = obs_stats[0]
            self.obs_rms.var[0] = obs_stats[1]**2
        self.rew_rms = RunningMeanStd()
        self.rew_rff = RewardForwardFilter(gamma)
        self.feature_size = 512
        self.conv_feature_size = 7*7*64

        # Learned predictor model
        self.forward_model = nn.Sequential(
                                            nn.Conv2d(
                                                in_channels=1,
                                                out_channels=32,
                                                kernel_size=8,
                                                stride=4),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=32,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=64,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1),
                                            nn.LeakyReLU(),
                                            Flatten(),
                                            nn.Linear(self.conv_feature_size, self.feature_size),
                                            nn.ReLU(),
                                            nn.Linear(self.feature_size, self.feature_size),
                                            nn.ReLU(),
                                            nn.Linear(self.feature_size, self.feature_size)
                                            )

        for param in self.forward_model:
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, np.sqrt(2))
                param.bias.data.zero_()

        # Fixed weight target model
        self.target_model = nn.Sequential(
                                            nn.Conv2d(
                                                in_channels=1,
                                                out_channels=32,
                                                kernel_size=8,
                                                stride=4),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=32,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2),
                                            nn.LeakyReLU(),
                                            nn.Conv2d(
                                                in_channels=64,
                                                out_channels=64,
                                                kernel_size=3,
                                                stride=1),
                                            nn.LeakyReLU(),
                                            Flatten(),
                                            nn.Linear(self.conv_feature_size, self.feature_size)
                                        )

        for param in self.target_model:
            if isinstance(param, nn.Conv2d) or isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, np.sqrt(2))
                param.bias.data.zero_()
        for param in self.target_model.parameters():
            param.requires_grad = False


    def forward(self, obs, not_done=None):

        # in case of frame stacking
        obs = obs[:,:,-1,:,:]
        obs = obs.unsqueeze(2)
        obs_cpu = obs.clone().cpu().data.numpy()

        # img = np.squeeze(obs.data.numpy()[0][0])
        # mean = np.squeeze(self.obs_rms.mean)
        # var = np.squeeze(self.obs_rms.var)
        # std = np.squeeze(np.sqrt(self.obs_rms.var))
        # cv2.imwrite('images/original.png', img)
        # cv2.imwrite('images/mean.png', mean)
        # cv2.imwrite('images/var.png', var)
        # cv2.imwrite('images/std.png', std)
        # cv2.imwrite('images/whitened.png', img-mean)
        # cv2.imwrite('images/final.png', (img-mean)/std)
        # cv2.imwrite('images/scaled_final.png', ((img-mean)/std)*111)
        #print("Final", np.min(((img-mean)/std).ravel()), np.mean(((img-mean)/std).ravel()), np.max(((img-mean)/std).ravel()))
        # print("#"*100 + "\n")

        # for o in obs_cpu:
        #     o = np.squeeze(o)
        #     vals = set(o.ravel())
        #     mode = 'none'
        #     for i in vals:
        #         if 34. < i < 36.:
        #             mode = 'blue'
            
        #     if mode != 'none':
        #         num = len(os.listdir('data/{}'.format(mode)))+1
        #         cv2.imwrite('data/{}/{}.png'.format(mode, num), o)

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        # lead_dim is just number of leading dimensions: e.g. [T, B] = 2 or [] = 0.
        lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
        
        if self.device == torch.device('cuda:0'):
            obs_mean = torch.from_numpy(self.obs_rms.mean).float().cuda()
            obs_var = torch.from_numpy(self.obs_rms.var).float().cuda()
        else:
            obs_mean = torch.from_numpy(self.obs_rms.mean).float()
            obs_var = torch.from_numpy(self.obs_rms.var).float()
        norm_obs = (obs.clone().float() - obs_mean) / (torch.sqrt(obs_var)+1e-10)
        norm_obs = torch.clamp(norm_obs, min=-5, max=5).float()

        # prediction target
        phi = self.target_model(norm_obs.clone().detach().view(T * B, *img_shape)).view(T, B, -1)

        # make prediction
        predicted_phi = self.forward_model(norm_obs.detach().view(T * B, *img_shape)).view(T, B, -1)

        # update statistics
        if not_done is not None:
            valid_obs = np.reshape(obs_cpu[np.where(not_done==1)[:2]], (-1, self.c, self.h, self.w))
            self.obs_rms.update(valid_obs)

        return phi, predicted_phi, T

    def compute_bonus(self, next_observation, done):
        done = done.cpu().data.numpy()
        not_done = np.abs(done-1)
        phi, predicted_phi, T = self.forward(next_observation, not_done=not_done)
        rewards = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size

        # update running mean
        rewards_cpu = rewards.clone().cpu().data.numpy()
        total_rew_per_env = np.array([self.rew_rff.update(rewards_cpu[i], not_done=not_done[i]) for i in range(T)])
        self.rew_rms.update_from_moments(np.mean(total_rew_per_env), np.var(total_rew_per_env), np.sum(not_done))

        # normalize rewards
        if self.device == torch.device('cuda:0'):
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float().cuda()
            not_done = torch.from_numpy(not_done).float().cuda()
        else:
            rew_var = torch.from_numpy(np.array(self.rew_rms.var)).float()
            not_done = torch.from_numpy(not_done).float()
        rewards /= torch.sqrt(rew_var)

        # apply done mask
        rewards *= not_done
        return self.prediction_beta * rewards

    def compute_loss(self, next_observations, valid):
        phi, predicted_phi, _ = self.forward(next_observations, not_done=None)
        forward_loss = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size
        mask = torch.rand(forward_loss.shape)
        mask = 1.0 - (mask > self.drop_probability).float().to(self.device)
        net_mask = mask * valid
        forward_loss = torch.sum(forward_loss * net_mask.detach()) / torch.sum(net_mask.detach())
        return forward_loss


