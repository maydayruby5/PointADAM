import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from PointDA.data.dataloader import ScanNet, ModelNet, ShapeNet, label_to_idx
# from PointDA.Models_new import PointNet, DGCNN
from Models_2 import DGCLIP
from utils import pc_utils
from DefRec_and_PCM import DefRec, PCM
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import seaborn as sns
NWORKERS=4
MAX_LOSS = 9 * (10**9)


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='DefRec_PCM',  help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
# parser.add_argument('--src_dataset', type=str, default='modelnet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--src_dataset1', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--src_dataset2', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
# parser.add_argument('--src_dataset3', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='modelnet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=150, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--DefRec_on_src', type=str2bool, default=True, help='Using DefRec in source')
parser.add_argument('--apply_PCM', type=str2bool, default=True, help='Using mixup in source')
parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size', help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

args = parser.parse_args()

# ==================
# init
# ==================
io = utils.log.IOStream(args)
io.cprint(str(args))

random.seed(1)
np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')


# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


src_dataset1 = args.src_dataset1
src_dataset2 = args.src_dataset2
# src_dataset3 = args.src_dataset3


# src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset
data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

# src_trainset = data_func[src_dataset](io, args.dataroot, 'train')
src_trainset1 = data_func[src_dataset1](io, args.dataroot, 'train')
src_trainset2 = data_func[src_dataset2](io, args.dataroot, 'train')
trgt_trainset = data_func[trgt_dataset](io, args.dataroot, 'train')
trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test')

# Creating data indices for training and validation splits:
# src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
src_train_sampler1, src_valid_sampler1 = split_set(src_trainset1, src_dataset1, "source")
src_train_sampler2, src_valid_sampler2 = split_set(src_trainset2, src_dataset2, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")


# dataloaders for source and target
# src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
#                                sampler=src_train_sampler, drop_last=True)
# src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
#                              sampler=src_valid_sampler)

# src_train_loader1 = DataLoader(src_trainset1, num_workers=NWORKERS, batch_size=args.batch_size,
#                                sampler=src_train_sampler1, drop_last=True)
# src_val_loader1 = DataLoader(src_trainset1, num_workers=NWORKERS, batch_size=args.test_batch_size,
#                              sampler=src_valid_sampler1)
# src_train_loader2 = DataLoader(src_trainset2, num_workers=NWORKERS, batch_size=args.batch_size,
#                                sampler=src_train_sampler2, drop_last=True)
# src_val_loader2 = DataLoader(src_trainset2, num_workers=NWORKERS, batch_size=args.test_batch_size,
#                              sampler=src_valid_sampler2)


# trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
#                                 sampler=trgt_train_sampler, drop_last=True)
# trgt_val_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
#                                   sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

# ==================
# Init Model
# ==================
if args.model == 'pointnet':
    model = PointNet(args)
elif args.model == 'dgcnn':
    #model = DGCNN(args)
    model = DGCLIP(args)
else:
    raise Exception("Not implemented")

model = model.to(device)

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
# best_model = copy.deepcopy(model)

model.load_state_dict(torch.load('./experiments/model2shape_clip_0711/model.pt'))



def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):

    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_feature = []
        test_true = []
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            # device = torch.device("cuda:" + str(data.get_device()) if args.cuda else "cpu")
            # preds1, loss, logits, _, _, _, feature = model(data, labels, activate_DefRec=False)
            # logits, feature = model(data, labels, activate_DefRec=False)
            logits, feature = model(data, activate_DefRec=False)
            # print(out.shape)
            # loss = criterion(logits["cls"], labels)
            # print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits['cls'].max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            test_feature.append(feature.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_feature = np.concatenate(test_feature)
    print(test_true.shape)
    print(test_pred.shape)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat, test_true, test_feature

# # ==================
# # Init Model
# # ==================
# if args.model == 'pointnet':
#     model = PointNet(args)
#     model.load_state_dict(torch.load('./experiments/test_adv_new/model.pt'))
#     # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('./experiments/test_adv_new/model.pt')})
#
# elif args.model == 'dgcnn':
#     model = DGCNN(args)
#     model.load_state_dict(torch.load('./experiments/test_adv_new/model.pt'))
#     # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('./experiments/test_adv_new/model.pt')})
# else:
#     raise Exception("Not implemented")
#
# model = model.to(device)

#===================
# Test
#===================



# idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
#                 4: "chair", 5: "lamp", 6: "monitor",
#                 7: "plant", 8: "sofa", 9: "table"}

def scatter(x, colors):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=2, s=20, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}

def plot_embedding(data, label):
    # “data为n * 2
    # 矩阵，label为n * 1
    # 向量，对应着data的标签, title未使用”

    fig = plt.figure()
    ax = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []
    type8_x = []
    type8_y = []
    type9_x = []
    type9_y = []
    type10_x = []
    type10_y = []
    type11_x = []
    type11_y = []

    for i in range(data.shape[0]):
        if label[i] == 0:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])
        if label[i] == 1:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])
        if label[i] == 2:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])
        if label[i] == 3:
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])
        if label[i] == 4:
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])
        if label[i] == 5:
            type6_x.append(data[i][0])
            type6_y.append(data[i][1])
        if label[i] == 6:
            type7_x.append(data[i][0])
            type7_y.append(data[i][1])
        if label[i] == 7:
            type8_x.append(data[i][0])
            type8_y.append(data[i][1])
        if label[i] == 8:
            type9_x.append(data[i][0])
            type9_y.append(data[i][1])
        if label[i] == 9:
            type10_x.append(data[i][0])
            type10_y.append(data[i][1])
        if label[i] == 10:
            type11_x.append(data[i][0])
            type11_y.append(data[i][1])

    color = plt.cm.Set3(0)
    color = np.array(color).reshape(1, 4)
    color1 = plt.cm.Set3(1)
    color1 = np.array(color1).reshape(1, 4)
    color2 = plt.cm.Set3(2)
    color2 = np.array(color2).reshape(1, 4)
    color3 = plt.cm.Set3(3)
    color3 = np.array(color3).reshape(1, 4)

    type1 = plt.scatter(type1_x, type1_y, s=10, c='r')
    type2 = plt.scatter(type2_x, type2_y, s=10, c='g')
    type3 = plt.scatter(type3_x, type3_y, s=10, c='b')
    type4 = plt.scatter(type4_x, type4_y, s=10, c='k')
    type5 = plt.scatter(type5_x, type5_y, s=10, c='c')
    type6 = plt.scatter(type6_x, type6_y, s=10, c='m')
    type7 = plt.scatter(type7_x, type7_y, s=10, c='y')
    type8 = plt.scatter(type8_x, type8_y, s=10, c=color)
    type9 = plt.scatter(type9_x, type9_y, s=10, c=color1)
    type10 = plt.scatter(type10_x, type10_y, s=10, c=color2)
    # type11 = plt.scatter(type11_x, type11_y, s=10, c='r')
    plt.legend((type1, type2, type3, type4, type5, type6, type7, type8, type9, type10),
               ('bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp', 'monitor', 'plant', 'sofa', 'table'), fontsize=12,ncol=6,loc='lower left')

    # plt.xticks(np.linspace(int(x_min[0]), math.ceil(x_max[0]), 5))
    # plt.yticks(np.linspace(int(x_min[1]), math.ceil(x_max[1]), 5))
    plt.xticks()
    plt.yticks()
    # plt.title(title)


    # ax.spines['right'].set_visible(False)  # 去除右边框
    # ax.spines['top'].set_visible(False)  # 去除上边框
    return fig

trgt_test_acc, trgt_test_loss, trgt_conf_mat, label, feature = test(trgt_test_loader, model, "Target", "Test", 0)
# digits = load_digits()
X_tsne = TSNE(n_components=2,random_state=33).fit_transform(feature)
# X_pca = PCA(n_components=2).fit_transform(feature)

ckpt_dir="images"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# scatter(X_tsne, label)
plot_embedding(X_tsne,  label)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label,label="t-SNE")
# plt.legend()
# plt.subplot(122)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label,label="PCA")
# plt.legend()
# plt.savefig('images/modelnet_tsne-pca.png', dpi=120)
plt.savefig('images/shapenet2modelnetwodann_0501.png', dpi=300)
# plt.show()


#
#
# io.cprint("target test accuracy: %.4f" % (trgt_test_acc))
# io.cprint("Test confusion matrix:")
# io.cprint('\n' + str(trgt_conf_mat))
