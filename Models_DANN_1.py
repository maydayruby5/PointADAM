import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from torch.autograd import Function

K = 20


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    # (batch_size, num_points, k)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, args, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    # Run on cpu or gpu
    device = torch.device("cuda:" + str(x.get_device()) if args.cuda else "cpu")
    # device = torch.device("cuda:0")

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # matrix [k*num_points*batch_size,3]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', bias=True):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='relu', bias=True):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch, bias=bias),
                nn.BatchNorm1d(out_ch),
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return: Transformation matrix of size 3xK """

    def __init__(self, args, in_ch, out=3):
        super(transform_net, self).__init__()
        self.K = out
        self.args = args

        activation = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = False if args.model == 'dgcnn' else True

        self.conv2d1 = conv_2d(in_ch, 64, kernel=1, activation=activation, bias=bias)
        self.conv2d2 = conv_2d(64, 128, kernel=1, activation=activation, bias=bias)
        self.conv2d3 = conv_2d(128, 1024, kernel=1, activation=activation, bias=bias)
        self.fc1 = fc_layer(1024, 512, activation=activation, bias=bias, bn=True)
        self.fc2 = fc_layer(512, 256, activation=activation, bn=True)
        self.fc3 = nn.Linear(256, out * out)

    def forward(self, x):
        device = torch.device("cuda:" + str(x.get_device()) if self.args.cuda else "cpu")

        x = self.conv2d1(x)
        x = self.conv2d2(x)
        if self.args.model == "dgcnn":
            x = x.max(dim=-1, keepdim=False)[0]
            x = torch.unsqueeze(x, dim=3)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x


class PointNet(nn.Module):
    def __init__(self, args, num_class=10):
        super(PointNet, self).__init__()
        self.args = args

        self.trans_net1 = transform_net(args, 3, 3)
        self.trans_net2 = transform_net(args, 64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        num_f_prev = 64 + 64 + 64 + 128

        self.C = classifier(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)

    def forward(self, x, activate_DefRec=False):
        num_points = x.size(2)
        x = torch.unsqueeze(x, dim=3)

        logits = {}

        transform = self.trans_net1(x)
        x = x.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        transform = self.trans_net2(x2)
        x = x2.transpose(2, 1)
        x = x.squeeze(dim=3)
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1)
        x3 = self.conv3(x)
        x4 = self.conv4(x3)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x4)
        x5, _ = torch.max(x, dim=2, keepdim=False)
        x = x5.squeeze(dim=2)  # batchsize*1024

        logits["cls"] = self.C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat.squeeze(dim=3), x5.repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits


class DGCNN(nn.Module):
    def __init__(self, args, num_class=10):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = K

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.bn5 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)

        # self.Mi = Mine(args)

        self.C = classifier(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)

    def forward(self, x, activate_DefRec=False):
        batch_size = x.size(0)
        num_points = x.size(2)
        logits = {}

        x0 = get_graph_feature(x, self.args, k=self.k)
        transformd_x0 = self.input_transform_net(x0)
        x = torch.matmul(transformd_x0, x)

        x = get_graph_feature(x, self.args, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, self.args, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, self.args, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, self.args, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)

        x5 = F.leaky_relu(self.bn5(self.conv5(x_cat)), negative_slope=0.2)

        images = self.mv_proj(pc).type(self.dtype)

        with torch.no_grad():
            # Image features
            image_feat = self.visual_encoder(images)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            image_feat = image_feat.reshape(-1, self.num_views * self.channel)



        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        x5 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x = x5
       # print(x.shape)


        logits["cls"] = self.C(x)
        # logits["cls"] = self.C(x)
        # logits["dw"] = self.C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x5.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits, x

def mv_proj(self, pc):
    img = self.get_img(pc).cuda()
    img = img.unsqueeze(1).repeat(1, 3, 1, 1)
    img = torch.nn.functional.upsample(img, size=(224, 224), mode='bilinear', align_corners=True)
    return img

class GRL(nn.Module):
    def __init__(self):
        super(GRL, self).__init__()
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 4000

    # @staticmethod
    def forward(self, input):
        self.iter_num += 1
        return input * 1.0

    # @staticmethod
    def backward(self, gradOutput):
        coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter))
                         - (self.high - self.low) + self.low)
        return -coeff * gradOutput


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0.0)
        self.fc2 = nn.Linear(512, 2)
        self.fc2.weight.data.normal_(0, 0.3)
        self.fc2.bias.data.fill_(0.0)
        self.ad_net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5),
            self.fc2
        )
        self.grl_layer = GRL()

    def forward(self, feature):
        adversarial_out = self.ad_net(self.grl_layer(feature))
        return adversarial_out




class classifier(nn.Module):
    def __init__(self, args, num_class=10):
        super(classifier, self).__init__()

        activate = 'leakyrelu' if args.model == 'dgcnn' else 'relu'
        bias = True if args.model == 'dgcnn' else False

        self.mlp1 = fc_layer(1024, 512, bias=bias, activation=activate, bn=True)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.mlp2 = fc_layer(512, 256, bias=True, activation=activate, bn=True)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.mlp3 = nn.Linear(256, num_class)

    def forward(self, x):
        x = self.dp1(self.mlp1(x))
        x2 = self.dp2(self.mlp2(x))
        logits = self.mlp3(x2)
        return logits


class RegionReconstruction(nn.Module):
    """
    Region Reconstruction Network - Reconstruction of a deformed region.
    For more details see https://arxiv.org/pdf/2003.12641.pdf
    """

    def __init__(self, args, input_size):
        super(RegionReconstruction, self).__init__()
        self.args = args
        self.of1 = 256
        self.of2 = 256
        self.of3 = 128

        self.bn1 = nn.BatchNorm1d(self.of1)
        self.bn2 = nn.BatchNorm1d(self.of2)
        self.bn3 = nn.BatchNorm1d(self.of3)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.dp2 = nn.Dropout(p=args.dropout)

        self.conv1 = nn.Conv1d(input_size, self.of1, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(self.of1, self.of2, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(self.of2, self.of3, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(self.of3, 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dp1(F.relu(self.bn1(self.conv1(x))))
        x = self.dp2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x.permute(0, 2, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DA on Point Clouds')
    parser.add_argument('--exp_name', type=str, default='DefRec_PCM', help='Name of the experiment')
    parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
    parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
    # parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--src_dataset1', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--src_dataset2', type=str, default='modelnet', choices=['modelnet', 'shapenet', 'scannet'])
    # parser.add_argument('--src_dataset3', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
    parser.add_argument('--epochs', type=int, default=150, help='number of episode to train')
    parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
    parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                        choices=['volume_based_voxels', 'volume_based_radius'],
                        help='distortion of points')
    parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of train batch per domain')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of test batch per domain')
    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
    parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
    parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    args = parser.parse_args()
    # args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
    print(torch.__version__)
    input = torch.autograd.Variable(torch.Tensor(16, 3, 1024))
    model = DGCNN(args)
    out = model(input)
    print(out.shape)