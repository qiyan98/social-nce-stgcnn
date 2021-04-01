import math
import torch


class EventSampler():
    '''
    Different sampling strategies for social contrastive learning
    '''

    def __init__(self, num_boundary=0, max_range=2.0, ratio_boundary=0.5, device='cpu'):
        # tunable param
        self.max_range = max_range
        self.num_boundary = num_boundary
        self.ratio_boundary = ratio_boundary
        # fixed param
        self.noise_local = 0.05
        self.min_separation = 0.2       # env-dependent parameter, diameter of agents
        self.max_separation = 2.5       # env-dependent parameter, diameter of agents
        self.agent_zone = self.min_separation * torch.tensor([
            [1.0, 0.0], [-1.0, 0.0],
            [0.0, 1.0], [0.0, -1.0],
            [0.707, 0.707], [0.707, -0.707],
            [-0.707, 0.707], [-0.707, -0.707]], device=device)        # regional surroundings
        self.device = device

    def _valid_check(self, pos_seed, neg_seed):
        '''
        Check validity of sample seeds, mask the frames that are invalid at the end of episodes
        '''
        dim_seed = len(pos_seed.shape)
        dist = (neg_seed - pos_seed.unsqueeze(dim_seed-1)).norm(dim=dim_seed)
        mask_valid = (dist > self.min_separation) & (dist < self.max_separation)
        # print('Ratio of valid data: {:.1f}%'.format(100 * mask_valid.sum().item() / mask_valid.numel()))  # debug
        return mask_valid

    def local_sampling(self, robot, pos_seed, neg_seed):
        '''
        Draw negative samples that are distant from the neighborhood of the postive sample
        '''

        mask_valid = self._valid_check(pos_seed, neg_seed)

        # positive samples
        sample_pos = pos_seed + torch.rand(pos_seed.size(), device=self.device).sub(0.5) * self.noise_local - robot[:, :2]

        # negative samples
        if self.num_boundary < 1:
            self.num_boundary = max(1, self.num_boundary)               # min value
            print("Warning: minimum number of negative")

        radius = torch.rand(pos_seed.size(0), self.num_boundary * 10, device=self.device) * self.max_range + self.min_separation
        theta = torch.rand(pos_seed.size(0), self.num_boundary * 10, device=self.device) * 2 * math.pi
        x = radius * torch.cos(theta) + radius * torch.sin(theta)
        y = radius * torch.sin(theta) - radius * torch.cos(theta)
        sample_neg = torch.cat([x.unsqueeze(2), y.unsqueeze(2)], axis=2) + pos_seed.unsqueeze(1)
        sample_neg += torch.rand(sample_neg.size(), device=self.device).sub(0.5) * self.noise_local - robot[:, None, :2]

        return sample_pos, sample_neg, mask_valid

    def event_sampling(self, robot, pos_seed, neg_seed):
        '''
        Draw negative samples based on regions of other agents across multiple time steps
        '''

        mask_valid = self._valid_check(pos_seed, neg_seed)

        # neighbor territory
        sample_territory = neg_seed[:, :, :, None, :] + self.agent_zone[None, None, None, :, :]
        sample_territory = sample_territory.view(sample_territory.size(0), sample_territory.size(1), sample_territory.size(2) * sample_territory.size(3), 2)

        # extend mask accordingly
        mask_valid = torch.stack([mask_valid]* len(self.agent_zone), dim=-1).view(*sample_territory.shape[:3])

        # primary-neighbor boundary
        if self.num_boundary > 0:
            alpha_list = torch.linspace(self.ratio_boundary, 1.0, steps=self.num_boundary)
            sample_boundary = []
            for alpha in alpha_list:
                sample_boundary.append(neg_seed * alpha + pos_seed.unsqueeze(2) * (1-alpha))
            sample_boundary = torch.cat(sample_boundary, axis=2)
            sample_neg = torch.cat([sample_boundary, sample_territory], axis=2)
        else:
            sample_neg = sample_territory

        # samples
        sample_pos = pos_seed + torch.rand(pos_seed.size(), device=self.device).sub(0.5) * self.noise_local - robot[:, None, :2]
        sample_neg += torch.rand(sample_neg.size(), device=self.device).sub(0.5) * self.noise_local - robot[:, None, None, :2]

        return sample_pos, sample_neg, mask_valid
