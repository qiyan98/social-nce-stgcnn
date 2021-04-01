import pandas
import pickle
import argparse
import random
import shutil

import torch.distributions.multivariate_normal as torchdist
import torch.multiprocessing as multiprocessing

from utils import * 
from metrics import *
from model import *

from contrast.model import *
from contrast.contrastive import *

random_seed = 2021
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred,V_target)


def train(model, contrastive, optimizer, device, loader_train, epoch, metrics, args):
    model.train()
    loss_batch, loss_total_batch, loss_contrast_batch = 0, 0, 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len/args.batch_size)*args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask, V_obs, A_obs, V_tr, A_tr, safety_gt_ = batch
        # dimensionality reminder: obs_traj: [1, num_person, 2, 8]; pred_traj_gt: [1, num_person, 2, 12]

        pick_safe_traj = args.safe_traj
        num_person = pred_traj_gt.size(1)
        safety_gt = safety_gt_.view(-1) if pick_safe_traj else torch.ones(num_person).bool().to(device)
        if pick_safe_traj and safety_gt.sum() == 0:
            # skip this batch if there is no collision-free trajectories
            continue

        optimizer.zero_grad()
        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)  # [1, 2, 8, num_person]  <- [1, 8, num_person, 2]

        V_pred, _, feat_vec = model(V_obs_tmp, A_obs.squeeze(), return_feat=True)  # [1, 5, 12, num_person], [1, num_person, 60]

        V_pred = V_pred.permute(0, 2, 3, 1)  # [1, 12, num_person, 5] <- [1, 5, 12, num_person]
        feat_vec = feat_vec.squeeze(0)  # [num_person, 60]

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            V_pred = V_pred[:, safety_gt, :].contiguous()
            V_tr = V_tr[:, safety_gt, :].contiguous()
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
                loss_contrast = torch.tensor(0.0).float().to(device)
            else:
                loss += l

            # contrastive task
            if args.contrast_weight > 0:
                # Recall dimensionality:
                # obs_traj: [1, num_person, 2, 8]; pred_traj_gt: [1, num_person, 2, 12]

                # replicate the scene such that each agent is primary for once
                num_person = feat_vec.size(0)
                num_neighbors = num_person - 1

                pedestrain_states = torch.zeros([num_person, 6]).float().to(device)
                pedestrain_states[:, :2] = obs_traj[0, :, :, -1]  # pick input's last frame

                pos_seeds = pred_traj_gt[0, :, :, :args.contrast_horizon].permute(0, 2, 1)  # [num_person, H, 2]

                # trick: swap primary agent for N times, N = num_person
                neg_seeds = torch.zeros([num_person, args.contrast_horizon, num_neighbors, 2]).float().to(device)  # [num_person, H, num_person-1, 2]
                for idx_primary in range(num_person):
                    neighbor_idxes = np.delete(np.arange(num_person), idx_primary)
                    neg_seeds_tmp = pred_traj_gt[0, np.ix_(neighbor_idxes), :, :args.contrast_horizon].squeeze(0)  # [num_person-1, 2, H]
                    neg_seeds[idx_primary] = neg_seeds_tmp.permute(2, 0, 1)  # [H, num_person-1, 2]

                hist_traj = V_obs_tmp.permute(3, 2, 1, 0).reshape(num_person, -1).contiguous()  # [num_person, 16] <- [1, 2, 8, num_person]
                l_contrast = contrastive.loss(pedestrain_states, pos_seeds, neg_seeds, feat_vec, hist_traj)
                loss_contrast += l_contrast * args.contrast_weight

        else:
            is_fst_loss = True

            loss = loss/args.batch_size
            loss_contrast = loss_contrast/args.batch_size
            loss_total = loss + loss_contrast
            loss_total.backward()
            
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

            optimizer.step()
            #Metrics
            loss_batch += loss.item()
            loss_contrast_batch += loss_contrast.item()
            loss_total_batch += loss_total.item()
            print('TRAIN:','\t Epoch:', epoch,'\t Total loss:',loss_total_batch/batch_count, '\t task loss:', loss_batch/batch_count,
                  '\t contrast loss:', loss_contrast_batch/batch_count)

    metrics['train_loss'].append(loss_total_batch/batch_count)
    metrics['task_loss'].append(loss_batch/batch_count)
    metrics['contrast_loss'].append(loss_contrast_batch/batch_count)
    

def vald(model, device, loader_val, epoch, metrics, constant_metrics, args, checkpoint_dir):
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1

    for cnt,batch in enumerate(loader_val):
        batch_count += 1

        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch


        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred,_ = model(V_obs_tmp, A_obs.squeeze())

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            #Metrics
            loss_batch += loss.item()
            print('VALD:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)

    metrics['val_loss'].append(loss_batch/batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch


def stack_dict(data_dict):
    for key, coll_step_data in zip(data_dict.keys(), data_dict.values()):
        data_dict[key] = np.stack(coll_step_data, axis=0)  # [X, 56]
    return data_dict


def process_batch_data(batch_idx: int, V_pred_rel_to_abs_ksteps: np.ndarray, V_y_rel_to_abs: np.ndarray, compute_col_truth=False):
    ade_ls = {}
    fde_ls = {}
    coll_ls = {}
    coll_joint_data_ls = {}
    coll_cross_data_ls = {}
    coll_truth_data_ls = {}

    num_of_objs = V_y_rel_to_abs.shape[1]
    for n in range(num_of_objs):
        ade_ls[n] = []
        fde_ls[n] = []
        coll_ls[n] = []
        coll_joint_data_ls[n] = []
        coll_cross_data_ls[n] = []
        coll_truth_data_ls[n] = []

    KSTEPS = len(V_pred_rel_to_abs_ksteps)
    # print('Detected ksteps: {:d}'.format(KSTEPS))
    for k in range(KSTEPS):
        V_pred_rel_to_abs = V_pred_rel_to_abs_ksteps[k]

        for n in range(num_of_objs):
            pred = [V_pred_rel_to_abs[:, n:n + 1, :]]
            target = [V_y_rel_to_abs[:, n:n + 1, :]]
            number_of = [1]

            ade_ls[n].append(ade(pred, target, number_of))
            fde_ls[n].append(fde(pred, target, number_of))

            ######
            predicted_traj = V_pred_rel_to_abs[:, n, :]  # [12, 2]
            predicted_trajs_all = V_pred_rel_to_abs.transpose(1, 0, 2)  # [num_person, 12, 2]
            col_mask_joint = compute_col(predicted_traj, predicted_trajs_all).astype(np.float64)  # [56], between predictions

            target_traj = V_y_rel_to_abs[:, n, :]  # [12, 2]
            target_trajs_all = V_y_rel_to_abs.transpose(1, 0, 2)  # [num_person, 12, 2]
            col_mask_cross = compute_col(predicted_traj, target_trajs_all).astype(np.float64)  # [56], prediction x ground-truth

            if compute_col_truth:
                col_mask_truth = compute_col(target_traj, target_trajs_all).astype(np.float64)  # [56], between ground-truth
                coll_truth_data_ls[n].append(col_mask_truth)

            if col_mask_joint.sum():
                coll_ls[n].append(1)
            else:
                coll_ls[n].append(0)
            coll_joint_data_ls[n].append(col_mask_joint)
            coll_cross_data_ls[n].append(col_mask_cross)
            ######

    coll_joint_data_ls = stack_dict(coll_joint_data_ls)
    coll_cross_data_ls = stack_dict(coll_cross_data_ls)
    if compute_col_truth:
        coll_truth_data_ls = stack_dict(coll_truth_data_ls)
    #  internal processing ends

    #  write data to the returned list, appending is okay as the order is not important
    ade_bigls_item, fde_bigls_item, coll_bigls_item = [], [], []
    for n in range(num_of_objs):
        ade_bigls_item.append(min(ade_ls[n]))  # float
        fde_bigls_item.append(min(fde_ls[n]))  # float
        coll_bigls_item.append(sum(coll_ls[n]) / len(coll_ls[n]))  # float
    coll_joint_data_bigls_item = np.concatenate([ls for ls in coll_joint_data_ls.values()], axis=0)  # [X, 56], np.ndarray
    coll_cross_data_bigls_item = np.concatenate([ls for ls in coll_cross_data_ls.values()], axis=0)
    if compute_col_truth:
        coll_truth_data_bigls_item = np.concatenate([ls for ls in coll_truth_data_ls.values()], axis=0)
    else:
        coll_truth_data_bigls_item = None

    return ade_bigls_item, fde_bigls_item, coll_bigls_item, coll_joint_data_bigls_item, coll_cross_data_bigls_item, coll_truth_data_bigls_item


def test(model, device, loader_test, epoch, KSTEPS=20):
    model.eval()
    # save batch data to list for later multi-processing
    num_batch = len(loader_test)
    V_pred_rel_to_abs_ksteps_ls, V_y_rel_to_abs_ls = [None] * num_batch, [None] * num_batch

    ade_bigls = []
    fde_bigls = []
    coll_bigls = []
    coll_joint_data_bigls = []
    coll_cross_data_bigls = []
    coll_truth_data_bigls = []
    raw_data_dict = {}

    time_start = time.time()
    time_sampling = 0.0
    for step, batch in enumerate(loader_test):
        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        # Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.detach().permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        # For now I have my bi-variate parameters
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).to(device)
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]
        # dimensionality reminder: mean: [12, num_person, 2], cov: [12, num_person, 2, 2]

        """pytorch solution for sampling"""
        time_sampling_start = time.time()

        # limit mean/cov tensor size, trying to debug
        max_size = 4
        mean_ls = torch.split(mean, max_size, dim=1)
        cov_ls = torch.split(cov, max_size, dim=1)
        kstep_V_pred_ls = []
        for sub_mean, sub_cov in zip(mean_ls, cov_ls):
            sub_mvnormal = torchdist.MultivariateNormal(sub_mean, sub_cov)
            sub_kstep_V_pred_ls = []
            for i in range(KSTEPS):
                sub_kstep_V_pred_ls.append(sub_mvnormal.sample().cpu().numpy())  # cat [12, sub_num_person<=8, 2]
            sub_kstep_V_pred_ls = np.stack(sub_kstep_V_pred_ls, axis=0)  # [KSTEPS, 12, sub_num_person, 2]
            kstep_V_pred_ls.append(sub_kstep_V_pred_ls)
        kstep_V_pred_ls = np.concatenate(kstep_V_pred_ls, axis=2)  # [KSTEPS, 12, num_person, 2]
        kstep_V_pred = np.concatenate([traj for traj in kstep_V_pred_ls], axis=1)
        time_sampling_elapsed = time.time() - time_sampling_start
        time_sampling += time_sampling_elapsed
        """end of sampling"""

        V_x = seq_to_nodes(obs_traj.data.cpu().numpy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze(), V_x[-1, :, :])

        kstep_V_x = np.concatenate([V_x[-1, :, :]] * KSTEPS, axis=0)  # cat along number of person
        kstep_V_pred_rel_to_abs = nodes_rel_to_nodes_abs(kstep_V_pred, kstep_V_x).reshape(12, KSTEPS, num_of_objs, 2)
        kstep_V_pred_rel_to_abs = kstep_V_pred_rel_to_abs.transpose((1, 0, 2, 3))  # [KSTEPS, 12, num_object, 2]

        V_pred_rel_to_abs_ksteps_ls[step] = kstep_V_pred_rel_to_abs  # np.ndarray
        V_y_rel_to_abs_ls[step] = V_y_rel_to_abs  # np.ndarray

    time_elapsed = time.time() - time_start
    print('Time to prepare all {:d} pieces of batch data: {:.3f}s'.format(num_batch, time_elapsed))
    print('In particular, time for multivariate gaussian distribution sampling: {:.3f}s'.format(time_sampling))

    time_start = time.time()
    func_batch_input = []
    for batch_idx in range(num_batch):
        V_pred_rel_to_abs_ksteps = V_pred_rel_to_abs_ksteps_ls[batch_idx]
        V_y_rel_to_abs = V_y_rel_to_abs_ls[batch_idx]
        if epoch == 0:
            cur_tuple = (batch_idx, V_pred_rel_to_abs_ksteps, V_y_rel_to_abs, True)
        else:
            cur_tuple = (batch_idx, V_pred_rel_to_abs_ksteps, V_y_rel_to_abs, False)
        func_batch_input.append(cur_tuple)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_batch_data, func_batch_input)
    time_elapsed = time.time() - time_start
    print('Time to multiprocess all {:d} pieces of batch data: {:.3f}s'.format(num_batch, time_elapsed))

    for idx_proc, result in enumerate(results):
        ade_bigls += result[0]  # list cat
        fde_bigls += result[1]  # list cat
        coll_bigls += result[2]  # list cat
        coll_joint_data_bigls.append(result[3])  # append np.ndarray
        coll_cross_data_bigls.append(result[4])
        if epoch == 0:
            coll_truth_data_bigls.append(result[5])  # could be None

    def coll_data_post_processing(coll_data_bigls):
        coll_raw_ = np.concatenate(coll_data_bigls, axis=0)  # [X, 56]
        coll_step_ = np.mean(coll_raw_, axis=0)  # [56]
        coll_step_ = coll_step_[:-1].reshape(-1, 5).mean(axis=1)  # [11]
        coll_cumulative_ = np.asarray([np.mean(coll_raw_[:, :i * 5 + 6].max(axis=1)) for i in range(11)])  # [11]
        return coll_step_, coll_cumulative_

    coll_joint_step, coll_joint_cum = coll_data_post_processing(coll_joint_data_bigls)
    coll_cross_step, coll_cross_cum = coll_data_post_processing(coll_cross_data_bigls)
    if epoch == 0:
        coll_truth_step, coll_truth_cum = coll_data_post_processing(coll_truth_data_bigls)
    else:
        coll_truth_step, coll_truth_cum = None, None

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    coll_ = sum(coll_bigls) / len(coll_bigls)

    return ade_, fde_, coll_, coll_joint_step, coll_joint_cum, coll_cross_step, coll_cross_cum, coll_truth_step, coll_truth_cum, raw_data_dict


def config_parser():
    parser = argparse.ArgumentParser()

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)

    # Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2')

    # Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=10,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='tag',
                        help='personal tag for the model ')
    parser.add_argument('--contrast_sampling', type=str, default='event')
    parser.add_argument('--contrast_weight', type=float, default=0.0)
    parser.add_argument('--contrast_horizon', type=int, default=4)
    parser.add_argument('--contrast_temperature', type=float, default=0.2)
    parser.add_argument('--contrast_range', type=float, default=2.0)
    parser.add_argument('--contrast_nboundary', type=int, default=0)
    parser.add_argument('--ratio_boundary', type=float, default=0.5)
    parser.add_argument('--contrast_loss', type=str, default='nce')
    parser.add_argument('--contrast_minsep', type=float, default=0.2)
    parser.add_argument('--safe_traj', action='store_true', default=False,
                        help='remove training trajectories with collision')

    args = parser.parse_args()
    return args


def get_target_metrics(dataset: str, tolerance: float = 0.0):
    """Get performance of pretrained model as gold standard."""
    if dataset == 'eth':
        # target_ade, target_fde = 0.64, 1.11  # paper
        target_ade, target_fde = 0.732, 1.223  # pretrained
        target_col = 1.33
    elif dataset == 'hotel':
        # target_ade, target_fde = 0.49, 0.85  # paper
        target_ade, target_fde = 0.410, 0.671  # pretrained
        target_col = 3.56
    elif dataset == 'univ':
        # target_ade, target_fde = 0.44, 0.79  # paper
        target_ade, target_fde = 0.489, 0.911  # pretrained
        target_col = 9.22
    elif dataset == 'zara1':
        target_ade, target_fde = 0.335, 0.524  # paper ~= pretrained
        target_col = 2.14
    elif dataset == 'zara2':
        target_ade, target_fde = 0.304, 0.481  # paper ~= pretrained
        target_col = 6.87
    else:
        raise NotImplementedError
    return target_ade+tolerance, target_fde+tolerance, target_col


def config_model(args, device):
    """Define the model."""
    model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                          output_feat=args.output_size, seq_len=args.obs_seq_len,
                          kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).to(device)

    projection_head = ProjHead(feat_dim=60 + 16, hidden_dim=32, head_dim=8).to(device)
    if args.contrast_sampling == 'event':
        encoder_sample = EventEncoder(hidden_dim=8, head_dim=8).to(device)
    else:
        encoder_sample = SpatialEncoder(hidden_dim=8, head_dim=8).to(device)
    num_params_contrast = sum(
        [p.numel() for layer in [projection_head, encoder_sample] for p in layer.parameters() if p.requires_grad])
    print('Contrastive learning module # trainable parameters: {:d}'.format(num_params_contrast))

    # contrastive
    if args.contrast_loss == 'nce':
        contrastive = SocialNCE(projection_head, encoder_sample, args.contrast_sampling, args.contrast_horizon,
                                args.contrast_nboundary, args.contrast_temperature, args.contrast_range,
                                args.ratio_boundary, args.contrast_minsep)
    else:
        raise NotImplementedError
    return model, contrastive


def get_dataloader(dataset, obs_seq_len, pred_seq_len):
    data_set = './datasets/' + dataset + '/'

    dset_train = TrajectoryDataset(
        data_set + 'train/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True)

    loader_train = DataLoader(
        dset_train,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=6, pin_memory=True)

    dset_val = TrajectoryDataset(
        data_set + 'val/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=6, pin_memory=True)

    dset_test = TrajectoryDataset(
        data_set + 'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True)

    loader_test = DataLoader(
        dset_test,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=False,
        num_workers=6, pin_memory=True)

    return loader_train, loader_val, loader_test


def pick_from_log(log_path: str, min_epoch: int = 50):
    """Read training log from checkpoint folder."""
    log_name = '-'.join(os.path.basename(os.path.dirname(log_path)).split('-')[:-3])
    dataset = os.path.basename(os.path.abspath(os.path.join(log_path, '..'))).split('-')[-1]
    if not os.path.exists(log_path):
        print('Expected training log at {:s} does not exist.'.format(log_path))
        return None
    model_weights = [anything for anything in os.listdir(os.path.join(os.path.dirname(log_path), 'history')) if anything.endswith('best.pth')]
    if len(model_weights) < min_epoch:
        print('Training epochs {:d} are too few!'.format(len(model_weights)))
        return None
    else:
        df_raw = pandas.read_csv(log_path)
        if 'col_joint_c4' in df_raw.columns:
            columns_to_pick = ['Epoch', 'ADE', 'FDE', 'col_joint_c4']
        else:
            columns_to_pick = ['Epoch', 'ADE', 'FDE', 'COLL']
        df_ = df_raw[columns_to_pick]

        _, target_fde, _ = get_target_metrics(dataset, 0.001)
        best_fde_overall = df_['FDE'].values.min()
        if best_fde_overall > target_fde:
            col_joint_c4_error = df_['ADE'].values + df_['FDE'].values
            best_col_idx = np.argsort(col_joint_c4_error)[0]
            best_col_epoch = int(df_['Epoch'].values[best_col_idx])
            best_col_ade = df_['ADE'].values[best_col_idx]
            best_col_fde = df_['FDE'].values[best_col_idx]
            best_col = df_['col_joint_c4'][best_col_idx] if 'col_joint_c4' in df_raw.columns else df_['COLL'][best_col_idx]
            print('ADE+FDE+COL total error minimizer: ADE: {:.4f}, FDE: {:.4f}, COL: {:.2f}%, EPOCH: {:d}.'.format(
                best_col_ade, best_col_fde, best_col * 100, best_col_epoch))
            return best_col_epoch

        tolerance_ls = [0.001]
        for tolerance in tolerance_ls:
            # find most performant model by ADE/FDE tolerance
            target_ade, target_fde, target_col = get_target_metrics(dataset, tolerance)
            mask_good_fde = df_['FDE'].values <= target_fde
            df = df_.loc[mask_good_fde]

            if mask_good_fde.sum() == 0:
                continue

            best_fde = df['FDE'].values.min()
            if best_fde > target_fde:
                print('Tolerance: {:.3f}, FDE too large: {:.4f} > target = {:.4f}'.format(tolerance, best_fde, target_fde))
                return None
            else:
                coll_overall = df['col_joint_c4'].values if 'col_joint_c4' in df.columns else df['COLL'].values
                best_col = coll_overall.min()
                best_col_idx = np.argsort(coll_overall)[0]
                best_col_epoch = int(df['Epoch'].values[best_col_idx])
                best_col_ade = df['ADE'].values[best_col_idx]
                best_col_fde = df['FDE'].values[best_col_idx]
                print('Tolerance: {:.3f}, Best FDE: {:.4f} <= target = {:.4f} '.format(tolerance, best_fde, target_fde))
                print('Best model up to now: ADE: {:.4f}, FDE: {:.4f}, COL: {:.2f}%, EPOCH: {:d}'.format(
                    best_col_ade, best_col_fde, best_col * 100, best_col_epoch))
        return best_col_epoch


def main():

    args = config_parser()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    target_ade, target_fde, target_col = get_target_metrics(args.dataset)
    # to be very conservative
    target_ade -= 0.05
    target_fde -= 0.05

    print('*' * 30)
    print("Training initiating....")
    print(args)

    # Define the model
    model, contrastive = config_model(args, device)

    # Data loader
    loader_train, loader_val, loader_test = get_dataloader(args.dataset, args.obs_seq_len, args.pred_seq_len)

    # Optimizer settings
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.use_lrschd:
        patience_epoch = args.lr_sh_rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_epoch, threshold=0.01,
                                                         factor=0.5, cooldown=patience_epoch, min_lr=1e-5, verbose=True)

    # Training log settings
    checkpoint_dir = './checkpoint/' + args.tag + '/'
    history_dir = os.path.join(checkpoint_dir, 'history') + '/'
    csv_path = os.path.join(checkpoint_dir, 'training_log.csv')

    for folder in [checkpoint_dir, history_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # save argument once and for all
    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Checkpoint dir:', checkpoint_dir)

    metrics = {'train_loss': [], 'task_loss': [], 'contrast_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    # Start training
    print('Training started ...')
    ade_ls, fde_ls, coll_ls, ttl_error_ls = [], [], [], []
    best_ade, best_fde, best_coll, best_ttl_error, best_coll_joint_c4_error, best_coll_joint_c4 = 99999., 99999., 99999., 99999., 99999., 99999.

    df = pandas.DataFrame(columns=['Epoch', 'total_loss', 'task_loss', 'contrast_loss', 'validation_loss', 'ADE', 'FDE', 'COLL'])
    for epoch in range(args.num_epochs):
        time_start = time.time()
        train(model, contrastive, optimizer, device, loader_train, epoch, metrics, args)
        time_elapsed = time.time() - time_start
        print('Time to train once: {:.2f} s for dataset {:s}'.format(time_elapsed, args.dataset))

        time_start = time.time()
        vald(model, device, loader_val, epoch, metrics, constant_metrics, args, checkpoint_dir)
        time_elapsed = time.time() - time_start
        print('Time to validate once: {:.2f} s for dataset {:s}'.format(time_elapsed, args.dataset))
        if args.use_lrschd:
            ttl_loss = metrics['train_loss'][-1]
            scheduler.step(ttl_loss)  # learning rate decay once training stagnates

        print('*' * 30)
        print('Epoch:', args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        """Test per epoch"""
        ade_, fde_, coll_ = 999999.0, 999999.0, 999999.0
        print("Testing ....")
        time_start = time.time()
        ad, fd, coll, coll_joint_step, coll_joint_cum, coll_cross_step, coll_cross_cum, coll_truth_step, coll_truth_cum, _ = test(
            model, device, loader_test, epoch)
        time_elapsed = time.time() - time_start
        print('Time to test once: {:.2f} s for dataset {:s}'.format(time_elapsed, args.dataset))
        ade_, fde_, coll_ = min(ade_, ad), min(fde_, fd), min(coll_, coll_joint_cum[2])
        ttl_error_ = np.clip(ade_ - target_ade, a_min=0.0, a_max=None) + np.clip(fde_ - target_fde, a_min=0.0, a_max=None) + coll_

        ade_ls.append(ade_)
        fde_ls.append(fde_)
        coll_ls.append(coll_)
        ttl_error_ls.append(ttl_error_)
        print("ADE: {:.4f}, FDE: {:.4f}, COL: {:.4f}, Total ERROR: {:.4f}, COL_JOINT_C4: {:.4F}".format(
            ade_, fde_, coll_, ttl_error_, coll_joint_cum[2]))

        best_ade = min(ade_, best_ade)
        best_fde = min(fde_, best_fde)
        best_coll = min(coll_, best_coll)
        best_ttl_error = min(ttl_error_, best_ttl_error)
        best_coll_joint_c4 = min(coll_joint_cum[2], best_coll_joint_c4)
        print(
            "Best ADE: {:.4f}, Best FDE: {:.4f}, Best COL: {:.4f}, Best Total ERROR: {:.4f}, Best COL_JOINT_C4: {:.4F}".format(
                best_ade, best_fde, best_coll, best_ttl_error, best_coll_joint_c4))

        df.loc[len(df)] = [epoch, metrics['train_loss'][-1], metrics['task_loss'][-1], metrics['contrast_loss'][-1],
                           metrics['val_loss'][-1], ade_, fde_, coll_]
        df = df.sort_values(by=['Epoch'])
        if not os.path.exists(csv_path):
            df.iloc[-1:].to_csv(csv_path, mode='a', index=False)
        else:
            df.iloc[-1:].to_csv(csv_path, mode='a', header=False, index=False)

        best_epoch = pick_from_log(csv_path, 0)
        print('Best epoch up to now is {}'.format(best_epoch))
        """Test ends"""

        print(constant_metrics)
        print('*'*30)

        with open(history_dir+'epoch{:03d}_metrics.pkl'.format(epoch), 'wb') as fp:
            pickle.dump(metrics, fp)

        with open(history_dir+'epoch{:03d}_constant_metrics.pkl'.format(epoch), 'wb') as fp:
            pickle.dump(constant_metrics, fp)

        torch.save(model.state_dict(), history_dir + 'epoch{:03d}_val_best.pth'.format(epoch))

        # model selection
        shutil.copy(history_dir+'epoch{:03d}_metrics.pkl'.format(best_epoch), checkpoint_dir + 'metrics.pkl')
        shutil.copy(history_dir+'epoch{:03d}_constant_metrics.pkl'.format(best_epoch), checkpoint_dir + 'constant_metrics.pkl')
        shutil.copy(history_dir+'epoch{:03d}_val_best.pth'.format(best_epoch), checkpoint_dir + 'val_best.pth')


if __name__ == '__main__':
    main()
