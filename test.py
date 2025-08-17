import pickle
import glob
from torch.utils.data.dataloader import DataLoader
from utils import *
from metrics import *
from model import TrajectoryModel
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time


def compute_batch_metric(pred, gt):
    """Get ADE, FDE, TCC scores for each pedestrian"""
    # Calculate ADEs and FDEs
    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs = temp.mean(dim=1).min(dim=0)[0]
    FDEs = temp[:, -1, :].min(dim=0)[0]

    # Calculate TCCs
    pred_best = pred[temp[:, -1, :].argmin(dim=0), :, range(pred.size(2)), :]
    pred_gt_stack = torch.stack([pred_best, gt.permute(1, 0, 2)], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    return ADEs, FDEs, TCCs


def test(model, loader_test, KSTEPS=20):

    model.eval()
    ade_all, fde_all, tcc_all = [], [], []

    step =0
    pic_cnt = 0
    for batch in loader_test:
        step+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, V_tr = batch

        identity_spatial = torch.ones((V_obs.shape[1], V_obs.shape[2], V_obs.shape[2])) * torch.eye(
            V_obs.shape[2])
        identity_temporal = torch.ones((V_obs.shape[2], V_obs.shape[1], V_obs.shape[1])) * torch.eye(
            V_obs.shape[1])
        identity_spatial = identity_spatial.cuda()
        identity_temporal = identity_temporal.cuda()
        identity = [identity_spatial, identity_temporal]

        CVM = V_obs[:, -1:, :, 1:]

        V_pred = model(V_obs, identity, obs_traj)  # A_obs <8, #, #>

        V_pred = V_pred + CVM

        # V_pred = V_pred.squeeze()
        V_tr = V_tr.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
        V_pred_traj_gt = pred_traj_gt.permute(0, 3, 1, 2).squeeze(dim=0)

        ade_stack, fde_stack, tcc_stack = [], [], []

        # Evaluate trajectories
        V_absl = V_pred.cumsum(dim=1) + V_obs_traj[[-1], :, :]

        ADEs, FDEs, TCCs = compute_batch_metric(V_absl, V_pred_traj_gt)

        ade_stack.append(ADEs.detach().cpu().numpy())
        fde_stack.append(FDEs.detach().cpu().numpy())
        tcc_stack.append(TCCs.detach().cpu().numpy())

        ade_all.append(np.array(ade_stack))
        fde_all.append(np.array(fde_stack))
        tcc_all.append(np.array(tcc_stack))

    ade_all = np.concatenate(ade_all, axis=1)
    fde_all = np.concatenate(fde_all, axis=1)
    tcc_all = np.concatenate(tcc_all, axis=1)

    mean_ade, mean_fde, mean_tcc = ade_all.mean(axis=0).mean(), fde_all.mean(axis=0).mean(), tcc_all.mean(axis=0).mean()
    return mean_ade, mean_fde, mean_tcc






def main():

    KSTEPS = 20
    ade_ls = []
    fde_ls = []
    print('Number of samples:', KSTEPS)
    print("*" * 50)
    root_ = './checkpoints/'
    dataset = [
        ### ETH UCY ###
        'GHGNN/eth',
        'GHGNN/hotel',
        'GHGNN/univ',
        'GHGNN/zara1',
        'GHGNN/zara2',
        ##### SDD #####
        'GHGNN/sdd',
    ]


    paths = list(map(lambda x: root_ + x, dataset))

    for feta in range(len(paths)):

        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:', exps)
        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + '/val_best.pth'
            args_path = exp_path + '/args.pkl'
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            # Data prep
            obs_seq_len = args.obs_len
            pred_seq_len = args.pred_len
            data_set = './dataset/' + args.dataset + '/'

            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1)

            loader_test = DataLoader(
                dset_test,
                batch_size=1,  # This is irrelative to the args batch size parameter
                shuffle=False,
                num_workers=1)

            model = TrajectoryModel(embedding_dims=64, number_gcn_layers=1, dropout=0.1,
                                    obs_len=8, pred_len=12, n_tcn=5).cuda()
            model.load_state_dict(torch.load(model_path))

            num_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters: {num_params}")

            ad_ = 999999
            fd_ = 999999
            print("Testing ....")
            ade_,fde_,raw_data_dict = test(model, loader_test)
            ade_ = min(ade_, ad_)
            fde_ = min(fde_, fd_)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ade:", ade_, " fde:", fde_)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params}")

        # 如果有GPU，则使用GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 计算显存占用（仅GPU有效）
        if device.type == 'cuda':
            # 获取当前显存使用情况（单位：字节）
            mem_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 转换为MB
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # 转换为MB
            print(f'GPU memory allocated: {mem_allocated:.2f} MB')
            print(f'GPU memory reserved: {mem_reserved:.2f} MB')
        else:
            print('Memory usage measurement is only available on GPU.')


        # 计算单次推理时间（inference time）
        input1 = torch.randn(1, 8, 2, 3)  # 按照你实际输入调整
        input1 = input1.to(device)

        identity_spatial = torch.ones((input1.shape[1], input1.shape[2], input1.shape[2])) * torch.eye(
            input1.shape[2])
        identity_temporal = torch.ones((input1.shape[2], input1.shape[1], input1.shape[1])) * torch.eye(
            input1.shape[1])
        identity_spatial = identity_spatial.cuda()
        identity_temporal = identity_temporal.cuda()
        input2 = [identity_spatial, identity_temporal]

        input3 = torch.randn(1, 2, 2, 8)  # 按照你实际输入调整
        input3 = input3.to(device)

        num_runs = 5
        with torch.no_grad():
            # Preheat 预热
            for _ in range(20):
                _ = model(input1, input2, input3)

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input1, input2, input3)
        end_time = time.time()

        avg_latency = (end_time - start_time) / num_runs  # 秒
        fps = 1.0 / avg_latency

        print(f'Average inference time per run: {avg_latency * 1000:.2f} ms')
        print(f'Frames per second (FPS): {fps:.2f}')

        print("*" * 50)


if __name__ == '__main__':
    main()