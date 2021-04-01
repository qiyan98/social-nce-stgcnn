import torch
import torch.nn as nn
from contrast.sampling import EventSampler
from contrast.visualize import plot_samples


class SocialNCE():
    '''
    Social contrastive loss, encourage the extracted motion representation to be aware of socially unacceptable events
    '''

    def __init__(self, head_projection=None, encoder_sample=None, sampling='social', horizon=3, num_boundary=0, temperature=0.07, max_range=2.0, ratio_boundary=0.5, contrast_minsep=0.2):
        # encoders
        self.head_projection = head_projection
        self.encoder_sample = encoder_sample
        # nce
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        # sampling
        self.sampling = sampling
        self.horizon = horizon
        self.device = next(head_projection.head.parameters()).device
        self.sampler = EventSampler(num_boundary, max_range, ratio_boundary, device=self.device)

    def loss(self, robot, pos, neg, feat, hist_traj=None):
        '''
        Input:
            robot: a tensor of shape (B, 6) for robot state, i.e. [x, y, vx, vy, gx, gy]
            pos: a tensor of shape (B, H, 2) for positive location seeds, i.e. [x, y], where H is the sampling horizion
            neg: a tensor of shape (B, H, N, 2) for negative location seeds
            feat: a tensor of shape (B, D) for extracted features, where D is the dimension of extracted motion features
        Output:
            social nce loss
        '''

        # sampling
        if self.sampling == 'local':
            sample_pos, sample_neg, mask_valid = self.sampler.local_sampling(robot, pos[:, self.horizon-1], neg[:, self.horizon-1])
        elif self.sampling == 'event':
            sample_pos, sample_neg, mask_valid = self.sampler.event_sampling(robot, pos, neg)
        else:
            raise NotImplementedError

        if self.sampling == 'event':
            # mask_valid should have the same shape as sample_neg, except for the last dimension
            assert mask_valid.shape == sample_neg.shape[0:3]

        # sanity check
        # self._sanity_check(robot, human, sample_pos, sample_neg)

        # obsv embedding, a tensor of shape (B, E) for motion embeddings
        if hist_traj is None:
            emb_obsv = self.head_projection(feat)
        else:
            emb_obsv = self.head_projection(torch.cat([feat, hist_traj], dim=1))
        query = nn.functional.normalize(emb_obsv, dim=1)

        if self.sampling == 'event':
            # event embedding
            time_pos = (torch.ones(sample_pos.size(0))[:, None] * (torch.arange(self.horizon) - (self.horizon-1.0)*(0.5))[None, :]).to(self.device)
            time_neg = (torch.ones(sample_neg.size(0), sample_neg.size(2))[:, None, :] * (torch.arange(self.horizon) - (self.horizon-1.0)*(0.5))[None, :, None]).to(self.device)
            emb_pos = self.encoder_sample(sample_pos, time_pos[:, :, None])
            emb_neg = self.encoder_sample(sample_neg, time_neg[:, :, :, None])
            # normalized embedding
            key_pos = nn.functional.normalize(emb_pos, dim=2)
            key_neg = nn.functional.normalize(emb_neg, dim=3)
            # similarity
            sim_pos = (query[:, None, :] * key_pos).sum(dim=2)
            sim_neg = (query[:, None, None, :] * key_neg).sum(dim=3)

            sim_neg.masked_fill_(~mask_valid, -10.0)  # set nan negatives to large negative value

            # logits
            logits = torch.cat([sim_pos.view(-1).unsqueeze(1), sim_neg.view(sim_neg.size(0), -1).repeat_interleave(self.horizon, dim=0)], dim=1) / self.temperature
        elif self.sampling == 'local':
            # sample embedding
            emb_pos = self.encoder_sample(sample_pos)
            emb_neg = self.encoder_sample(sample_neg)
            # normalized embedding
            key_pos = nn.functional.normalize(emb_pos, dim=1)
            key_neg = nn.functional.normalize(emb_neg, dim=2)
            # similarity
            sim_pos = (query * key_pos).sum(dim=1)
            sim_neg = (query[:, None, :] * key_neg).sum(dim=2)
            # logits
            logits = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1) / self.temperature
        else:
            raise NotImplementedError

        # loss
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        loss = self.criterion(logits, labels)

        return loss

    def _sanity_check(self, robot, human, pos, neg):
        '''
        Debug sampling strategy
        '''
        def tensor_to_array(in_var):
            if isinstance(in_var, torch.Tensor):
                return in_var.cpu()
            else:
                return in_var
        robot, human, pos, neg = tensor_to_array(robot), tensor_to_array(human), tensor_to_array(pos), tensor_to_array(neg)
        for i in range(len(robot)):
            if len(pos.shape) > 2:
                for k in range(self.horizon):
                    plot_samples(robot[i, :4], human[i], robot[i, 4:], pos[i, k], neg[i, k], fname='samples_{:d}_time_{:d}.png'.format(i, k))
            else:
                plot_samples(robot[i, :4], human[i], robot[i, 4:], pos[i], neg[i], fname='samples_{:d}.png'.format(i))


def plot_nce(primary, neighbor, positive, negative, fname=None, window=20):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(primary[:, 0], primary[:, 1], 'k-o')
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:, i, 0], neighbor[:, i, 1], 'b-o')
    ax.plot(positive[0], positive[1], 'gs', markerfacecolor='none', markersize=min(10, window))
    ax.plot(negative[:, 0], negative[:, 1], 'rx', markersize=min(10, window))
    ax.set_xlim(primary[-1, 0] - window, primary[-1, 0] + window)
    ax.set_ylim(primary[-1, 1] - window, primary[-1, 1] + window)
    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

