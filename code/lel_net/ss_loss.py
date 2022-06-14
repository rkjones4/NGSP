import torch
import torch.nn as nn
import torch.nn.functional as F


# Code from
# https://github.com/matheusgadelha/PointCloudLearningACD/blob/ba3348bf3b2aedcf6ee31a1053fb53302cab5a2c/models/pointnet_part_seg.py#L128

class get_selfsup_loss(nn.Module):
    def __init__(self, margin=0.5):
        super(get_selfsup_loss, self).__init__()
        self.margin = margin

    def forward(self, feat, target):
        
        feat = F.normalize(feat, p=2, dim=1)
        
        pair_sim = torch.bmm(feat.transpose(1,2), feat)
                
        one_hot_target = F.one_hot(target).float()
            
        pair_target = torch.bmm(one_hot_target, one_hot_target.transpose(1,2))        
                        
        cosine_loss = pair_target * (1. - pair_sim) + (1. - pair_target) * F.relu(pair_sim - self.margin)        
        
        diag_mask = 1 - torch.eye(cosine_loss.shape[-1])  # discard diag elems (always 1)

        with torch.no_grad():
            # balance positive and negative pairs
            pos_fraction = (pair_target.data == 1).float().mean()
            sample_neg = torch.cuda.FloatTensor(*pair_target.shape).uniform_() > 1 - pos_fraction
            sample_mask = (pair_target.data == 1) | sample_neg # all positives, sampled negatives

        cosine_loss = diag_mask.unsqueeze(0).cuda() * sample_mask.type(torch.cuda.FloatTensor) * cosine_loss
     
        total_loss = 0.5 * cosine_loss.mean() # scale down

        return total_loss
        
if __name__ == '__main__':
    loss = get_selfsup_loss()

    # batch, # feat, # points
    feat = torch.randn(2, 7, 13).cuda()
    # batch, points
    target = torch.randint(0, 5, (2, 13)).cuda()

    feat = torch.tensor([
        [
            [1., 0., 0.,0.],
            [0., 1., 0.,0.],
            [0.1,0.,0.,0.],
        ],
        [
            [0.1, 0., 0.,0.],
            [0.5,0.0,0.0,0.],
            [1.0,0.0,0.0,0.],
        ],
    ]).transpose(1,2).cuda()

    target = torch.tensor([
        [
            0,
            1,
            0,
        ],
        [
            0,
            1,
            0,
        ],
    ]).cuda()
        

    
    print(feat.shape)
    print(target.shape)
    
    l = loss(feat, target)

    print(l)

    
