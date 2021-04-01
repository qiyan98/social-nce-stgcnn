import pandas as pd
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import *
from metrics import *
from model import social_stgcnn
import copy
import random
import time


os.environ['KMP_DUPLICATE_LIB_OK'] ='True'  # debug

random_seed = 2021
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


def interpolate_traj(traj, num_interp=4):
    '''
    Add linearly interpolated points of a trajectory
    '''
    sz = traj.shape
    dense = np.zeros((sz[0], (sz[1] - 1) * (num_interp + 1) + 1, 2))
    dense[:, :1, :] = traj[:, :1]

    for i in range(num_interp+1):
        ratio = (i + 1) / (num_interp + 1)
        dense[:, i+1::num_interp+1, :] = traj[:, 0:-1] * (1 - ratio) + traj[:, 1:] * ratio

    return dense


def compute_col(predicted_traj, predicted_trajs_all, thres=0.2):
    '''
    Input:
        predicted_trajs: predicted trajectory of the primary agents, [12, 2]
        predicted_trajs_all: predicted trajectory of all agents in the scene, [num_person, 12, 2]
    '''
    ph = predicted_traj.shape[0]
    num_interp = 4
    assert predicted_trajs_all.shape[0] > 1

    dense_all = interpolate_traj(predicted_trajs_all, num_interp)
    dense_ego = interpolate_traj(predicted_traj[None, :], num_interp)
    distances = np.linalg.norm(dense_all - dense_ego, axis=-1)  # [num_person, 12 * num_interp]
    mask = distances[:, 0] > 0  # exclude primary agent itself
    return (distances[mask].min(axis=0) < thres)


def test(KSTEPS=20):
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    coll_bigls = []
    coll_step_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        num_of_objs = obs_traj_rel.shape[1]

        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.detach().permute(0, 2, 3, 1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat

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

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len

        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        coll_ls = {}
        coll_step_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []
            coll_ls[n] = []
            coll_step_ls[n] = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

                ######
                predicted_traj = V_pred_rel_to_abs[:, n, :]  # [12, 2]
                predicted_trajs_all = V_pred_rel_to_abs.copy().transpose(1, 0, 2)  # [num_person, 12, 2]

                col_mask = compute_col(predicted_traj, predicted_trajs_all).astype(np.float64)  # [56]
                if col_mask.sum():
                    coll_ls[n].append(1)
                else:
                    coll_ls[n].append(0)

                coll_step_ls[n].append(col_mask)
                ######

        for key, coll_step_data in zip(coll_step_ls.keys(), coll_step_ls.values()):
            coll_step_ls[key] = np.stack(coll_step_data, axis=0)  # [X, 56]

        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))
            coll_bigls.append(sum(coll_ls[n])/len(coll_ls[n]))

        coll_step_bigls.append(np.concatenate([ls for ls in coll_step_ls.values()], axis=0))  # [X, 56]

    coll_raw_ = np.concatenate(coll_step_bigls, axis=0)  # [X, 56]
    coll_step_ = np.mean(coll_raw_, axis=0)  # [56]
    coll_step_ = coll_step_[:-1].reshape(-1, 5).mean(axis=1)  # [11]
    coll_cumulative_ = np.asarray([np.mean(coll_raw_[:, :i * 5 + 6].max(axis=1)) for i in range(11)])  # [11]

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    coll_ = sum(coll_bigls) / len(coll_bigls)
    return ade_, fde_, coll_, coll_step_, coll_cumulative_, raw_data_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix tag for the model ')
    parser.add_argument('--tag', type=str, default='',
                        help='tag for csv path')
    parser.add_argument('--mode', type=str, default='fde',
                        help='metrics used to select model')
    ##Hos
    #############
    collision_thrshld = 0.2
    #############
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    opt = parser.parse_args()

    opt.prefix = opt.prefix[0] if isinstance(opt.prefix, list) else opt.prefix
    opt.tag = opt.tag[0] if isinstance(opt.tag, list) else opt.tag
    assert isinstance(opt.prefix, str)
    assert isinstance(opt.tag, str)
    opt.tag = 'default' if opt.tag == '' else opt.tag

    if opt.mode == 'snce':
        paths = sorted(['./checkpoint-snce/snce-social-stgcnn*'])
    elif opt.mode == 'baseline':
        paths = sorted(['./checkpoint-baseline/*social-stgcnn*'])
    elif opt.mode == 'random-sampling':
        paths = sorted(['./checkpoint-random-sampling/*social-stgcnn*'])
    else:
        paths = sorted(['./checkpoint/{:s}-social-stgcnn*'.format(opt.prefix) if opt.prefix is not '' else './checkpoint/social-stgcnn*'])
    KSTEPS=20

    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    df = pd.DataFrame(columns=['dataset', 'ade', 'fde', 'col', 'comment'])

    for feta in range(len(paths)):
        ade_ls = []
        fde_ls = []
        coll_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:', exps)

        for exp_path in exps:
            time_start = time.time()
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/best{:s}_val_best.pth'.format(opt.mode)
            stats = exp_path + '/best{:s}_constant_metrics.pkl'.format(opt.mode)
            if not os.path.exists(model_path):
                model_path = exp_path + '/val_best.pth'
                stats = exp_path + '/constant_metrics.pkl'
                print('Model weight for mode {:s} is not found. Use best validation model.'.format(opt.mode))
            else:
                print('Model weight for mode {:s} is loaded.'.format(opt.mode))
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            with open(stats, 'rb') as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            # Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = './datasets/' + args.dataset + '/'

            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1, norm_lap_matr=True)

            loader_test = DataLoader(
                dset_test,
                batch_size=1,  # This is irrelative to the args batch size parameter
                shuffle=False,
                num_workers=1)

            # Defining the model
            model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                                  output_feat=args.output_size, seq_len=args.obs_seq_len,
                                  kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            ade_ = 999999
            fde_ = 999999
            coll_ = 999999
            print("Testing ....")
            ad, fd, coll, coll_step, coll_cum, raw_data_dic_ = test()
            ade_ = min(ade_, ad)
            fde_ = min(fde_, fd)
            coll_ = min(coll_, coll_cum[2])  # use the coll_joint_cum up to step 4 as the collision metric
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            coll_ls.append(coll_)
            print("ADE:", ade_, " FDE:", fde_, " Coll:", coll_)

            df.loc[len(df)] = [args.dataset, ade_, fde_, coll_, 'model={:s}'.format(model_path)]
            time_elapsed = time.time() - time_start
            print('Elasped time: {:.2f} s'.format(time_elapsed))
        print("*" * 50)

        print("Avg ADE:", sum(ade_ls) / 5)
        print("Avg FDE:", sum(fde_ls) / 5)
        print("Avg coll:", sum(coll_ls) / 5)

        df = df.sort_values(by=['dataset'])

        df.loc[len(df)] = ['AVG', sum(ade_ls)/5, sum(fde_ls)/5, sum(coll_ls)/5, '']

        csv_path = 'results_{:s}.csv'.format(opt.tag)
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)
