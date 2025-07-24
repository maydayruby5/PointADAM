import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        # print(batch_size)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
#
#
# class CenterLoss(nn.Module):
#     def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True):
#         super(CenterLoss, self).__init__()
#         self.dim_hidden = dim_hidden
#         self.num_classes = num_classes
#         self.lambda_c = lambda_c
#         self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
#         self.use_cuda = use_cuda
#
#     def forward(self, y, hidden):
#         batch_size = hidden.size()[0]
#         expanded_centers = self.centers.index_select(dim=0, index=y)
#         intra_distances = hidden.dist(expanded_centers)
#         loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
#         return loss
#
#     def cuda(self, device_id=None):
#         """Moves all model parameters and buffers to the GPU.
#         Arguments:
#             device_id (int, optional): if specified, all parameters will be
#                 copied to that device
#         """
#         self.use_cuda = True
#         return self._apply(lambda t: t.cuda(device_id))
#
#
# def test():
#     ct = CenterLoss(1024, 10, use_cuda=False)
#     y = Variable(torch.LongTensor([0, 9, 6, 2]))
#     feat = Variable(torch.zeros(4, 1024), requires_grad=True)
#     print(feat.shape)
#
#     out = ct(y, feat)
#     print(out)
#     out.backward()
#
#
# if __name__ == '__main__':
#     test()
