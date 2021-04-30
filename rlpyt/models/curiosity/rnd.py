import os
from PIL import Image
import time
import numpy as np
import torch
from torch import nn
torch.set_printoptions(edgeitems=3)

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


    def forward(self, obs, done=None):
        a = time.perf_counter()
        # in case of frame stacking
        obs = obs[:,:,-1,:,:]
        obs = obs.unsqueeze(2)
        obs_cpu = obs.clone().cpu().data.numpy()

        b = time.perf_counter()
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

        c = time.perf_counter()
        # prediction target
        phi = self.target_model(norm_obs.clone().detach().view(T * B, *img_shape)).view(T, B, -1)
        
        d = time.perf_counter()
        # make prediction
        predicted_phi = self.forward_model(norm_obs.detach().view(T * B, *img_shape)).view(T, B, -1)
        e = time.perf_counter()

        # update statistics
        if done is not None:
            done = done.cpu().data.numpy()
            num_not_done = np.sum(np.abs(done-1), axis=0)
            obs_cpu = np.swapaxes(obs_cpu, 0, 1)
            valid_obs = obs_cpu[0][:int(num_not_done[0].item())]
            for i in range(1, B):
                obs_slice = obs_cpu[i][:int(num_not_done[i].item())]
                valid_obs = np.concatenate((valid_obs, obs_slice))
            self.obs_rms.update(valid_obs)
        f = time.perf_counter()
        print("Preproc: {}".format(b-a))
        print("ObsNorm: {}".format(c-b))
        print("Target: {}".format(d-c))
        print("Predict: {}".format(e-d))
        print("ObsUpdate: {}".format(f-e))
        print('-'*100)
        print("FORWARD TTL: {}".format(f-a))
        return phi, predicted_phi, T

    def compute_bonus(self, next_observation, done):
        x = time.perf_counter()
        phi, predicted_phi, T = self.forward(next_observation, done=done)
        rewards = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size

        # update running mean
        rewards_cpu = rewards.clone().cpu().data.numpy()
        not_done = torch.abs(done-1).cpu().data.numpy()
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
        z = time.perf_counter()
        print("BONUS TTL: {}".format(z-x))
        print('-'*100)
        return self.prediction_beta * rewards

    def compute_loss(self, next_observations, valid):
        l = time.perf_counter()
        phi, predicted_phi, _ = self.forward(next_observations, done=None)
        forward_loss = nn.functional.mse_loss(predicted_phi, phi.detach(), reduction='none').sum(-1)/self.feature_size
        mask = torch.rand(forward_loss.shape)
        mask = 1.0 - (mask > self.drop_probability).float().to(self.device)
        net_mask = mask * valid
        forward_loss = torch.sum(forward_loss * net_mask.detach()) / torch.sum(net_mask.detach())
        h = time.perf_counter()
        print("LOSS TTL: {}".format(h-l))
        print("-"*100)
        return forward_loss


