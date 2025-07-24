import torch
import torch.nn as nn
import torch.nn.functional as F

from PointDA.pointclip.co_attention import Self_Attention

from PointDA.pointclip.fewshot import PointCLIP_Model, load_clip_to_cpu
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

        activation = 'leakyrelu'
        bias = False

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
        if self.args.model == "dgcnn" or self.args.model == "dgclip":
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

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        x5 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x = x5


        logits["cls"] = self.C(x)

        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x5.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            logits["DefRec"] = self.DefRec(DefRec_input)

        return logits


class classifier(nn.Module):
    def __init__(self, args, num_class=10):
        super(classifier, self).__init__()

        activate = 'leakyrelu'
        bias = True

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


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            # Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_raw = self.mlp(x)# [8,2048]
        scale = F.sigmoid( channel_att_raw )# [8,2048]
        return x * scale

class dynamic_weight(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(dynamic_weight, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x):
        x_out = self.ChannelGate(x)# [8,2048]
        return x_out
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

class DGCLIP(nn.Module):
    def __init__(self, args, num_class=10):
        super(DGCLIP, self).__init__()
        self.args = args
        self.k = K

        self.clip_model = load_clip_to_cpu('RN50')
        self.imgF = PointCLIP_Model(self.clip_model)
        self.Self_Attention = Self_Attention(1024)
        self.dynamic_weight= dynamic_weight(2048)

        self.input_transform_net = transform_net(args, 6, 3)

        self.conv1 = conv_2d(6, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv2 = conv_2d(64 * 2, 64, kernel=1, bias=False, activation='leakyrelu')
        self.conv3 = conv_2d(64 * 2, 128, kernel=1, bias=False, activation='leakyrelu')
        self.conv4 = conv_2d(128 * 2, 256, kernel=1, bias=False, activation='leakyrelu')
        num_f_prev = 64 + 64 + 128 + 256

        self.bn5 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Conv1d(num_f_prev, 1024, kernel_size=1, bias=False)
        self.conv6 = nn.Sequential(
            nn.Linear(in_features=2048,out_features=1024),
            nn.ReLU())
        self.conv7 = nn.Sequential(
            nn.Linear(in_features=2048,out_features=1024),
            nn.ReLU())
        self.conv8 = nn.Sequential(
            nn.Linear(in_features=2048,out_features=1024),
            nn.ReLU())

        self.C = classifier(args, num_class)
        self.DefRec = RegionReconstruction(args, num_f_prev + 1024)


    def forward(self, x, activate_DefRec=False):
        # 获得clip图像特征
        imgF = self.imgF(x).float()# [8,1024]




        batch_size = x.size(0)
        num_points = x.size(2)
        cls_logits = {}

        x0 = get_graph_feature(x, self.args, k=self.k)# x[3,1024] x0[6,1024,20] 20个邻居
        transformd_x0 = self.input_transform_net(x0)# [3,3]
        x = torch.matmul(transformd_x0, x)# [3,1024]

        x = get_graph_feature(x, self.args, k=self.k)# [6,1024,20]
        x = self.conv1(x)# [64,1024,20]
        x1 = x.max(dim=-1, keepdim=False)[0]# [64,1024]

        x = get_graph_feature(x1, self.args, k=self.k)# [128,1024,20]
        x = self.conv2(x)# [64,1024,20]
        x2 = x.max(dim=-1, keepdim=False)[0]# [64,1024]

        x = get_graph_feature(x2, self.args, k=self.k)# [128,1024,20]
        x = self.conv3(x)# [128,1024,20]
        x3 = x.max(dim=-1, keepdim=False)[0]# [128,1024]

        x = get_graph_feature(x3, self.args, k=self.k)# [256,1024,20]
        x = self.conv4(x)# [8,256,1024,20]
        x4 = x.max(dim=-1, keepdim=False)[0]# [256,1024]

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)# [512,1024]

        x5 = F.leaky_relu(self.bn5(self.conv5(x_cat)), negative_slope=0.2)# [8,1024,1024]

        # Per feature take the point that have the highest (absolute) value.
        # Generate a feature vector for the whole shape
        x5 = F.adaptive_max_pool1d(x5, 1).view(batch_size, -1)
        x = x5 # [8,1024]
        if activate_DefRec:
            DefRec_input = torch.cat((x_cat, x5.unsqueeze(2).repeat(1, 1, num_points)), dim=1)
            cls_logits["DefRec"] = self.DefRec(DefRec_input)
        x_self = self.Self_Attention(x, imgF)

        #x_cat2=torch.cat((x,imgF),dim=1)# [8,2048]
        x_cat2 = torch.cat((x_self[0], x_self[1]), dim=1)  # [8,2048]

        x6=self.dynamic_weight(x_cat2)# [8,2048]
        x=self.conv6(x6)# [8,1024]
        # x = self.conv6(x_cat2)  # [8,1024]
        cls_logits["cls"] = self.C(x)



        return cls_logits

    def Flatten(self, x):
        return x.view(x.size(0), -1)
