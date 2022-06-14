import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2.pointnet2_utils as pointnet2_utils
import numpy as np
import sys

class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size, name=""):
        # type: (BatchNorm2d, int, AnyStr) -> None
        super(BatchNorm2d, self).__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)

class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        dilation,
        activation,
        bn,
        init,
        conv=None,
        norm_layer=None,
        bias=True,
        preact=False,
        name="",
    ):
        super(_ConvBase, self).__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = norm_layer(out_size)
            else:
                bn_unit = norm_layer(in_size)

        if preact:
            if bn:
                self.add_module(name + "normlayer", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "normlayer", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm2d,
    ):
        # type: (Conv2d, int, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr, _BNBase) -> None
        super(Conv2d, self).__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            norm_layer=norm_layer,
            bias=bias,
            preact=preact,
            name=name,
        )

class SharedMLP(nn.Sequential):
    def __init__(
        self,
        args,
        bn=False,
        activation=nn.ReLU(inplace=True),
        preact=False,
        first=False,
        name="",
    ):
        # type: (SharedMLP, List[int], bool, Any, bool, bool, AnyStr) -> None
        super(SharedMLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + "layer{}".format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0))
                    else None,
                    preact=preact,
                ),
            )

class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = SharedMLP(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)

class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(self, xyz, features=None):
        # type: (_PointnetSAModuleBase, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, N, C) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)

class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(SharedMLP(mlp_spec, bn=bn))

class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )

class PointNetPPSeg(nn.Module):
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        
        return xyz, features
    
    def __init__(self, num_classes, input_channels=0, use_xyz=True, USE_BN=False,DP=0.0):
        super(PointNetPPSeg, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[128 + input_channels, 128, 128, 128], bn = USE_BN)
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + 64, 256, 128], bn = USE_BN)            
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + 128, 256, 256], bn = USE_BN)            
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[512 + 256, 256, 256], bn = USE_BN)            
        )

        if USE_BN:
            self.fc_layer = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Dropout(DP),
                nn.Conv1d(128, num_classes, kernel_size=1),
            )

        else:
            self.fc_layer = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, bias=False),
                nn.ReLU(True),
                nn.Dropout(DP),
                nn.Conv1d(128, num_classes, kernel_size=1),
            )
            
    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0]).transpose(1,2)

class LelPointNetPPSeg(nn.Module):
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        
        return xyz, features
    
    def __init__(
            self, num_classes, input_channels, emb_dim,
            use_xyz=True, USE_BN=False,DP=0.0
    ):
        super(LelPointNetPPSeg, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[128 + input_channels, 128, 128, 128], bn = USE_BN)
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + 64, 256, 128], bn = USE_BN)            
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[256 + 128, 256, 256], bn = USE_BN)            
        )
        self.FP_modules.append(
            PointnetFPModule(mlp=[512 + 256, 256, 256], bn = USE_BN)            
        )

        self.emb_layer = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.ReLU(True),
        )
            
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Dropout(DP),
            nn.Conv1d(128, num_classes, kernel_size=1),
        )
            
    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        sem_out = self.fc_layer(l_features[0]).transpose(1,2)
        emb_out = self.emb_layer(l_features[0])

       
        return sem_out, emb_out

                
class PointNetPPEnc(nn.Module):
    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        
        return xyz, features
    
    def __init__(self, hidden_dim, input_channels, USE_BN):
        super(PointNetPPEnc, self).__init__()
        use_xyz=True
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, hidden_dim],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        
    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud)
        
        for i in range(len(self.SA_modules)):
            xyz, features = self.SA_modules[i](xyz, features)

        return features.squeeze(-1)

    
class PointNetPPCls(nn.Module):
    def _break_up_pc(self, pc, do_norm):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        
        if do_norm:
            with torch.no_grad():
                c = (xyz.max(dim=0).values + xyz.min(dim=0).values) / 2
                xyz -= c
                norm = torch.norm(xyz, dim=2).max(dim=1).values
                
                xyz /= (norm.view(-1,1,1) + 1e-8)
                            
        return xyz, features
    
    def __init__(self, num_classes, input_channels=0, USE_BN=True, do_norm=False, DP=0.0):
        super(PointNetPPCls, self).__init__()
        self.do_norm = do_norm
        use_xyz=True
        self.num_classes = num_classes
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                use_xyz=use_xyz,
                bn = USE_BN
            )
        )

        self.mlp = DMLP(1024, 256, 64, num_classes, DP)
        
    def forward(self, pointcloud):
        xyz, features = self._break_up_pc(pointcloud, self.do_norm)
        for i in range(len(self.SA_modules)):
            xyz, features = self.SA_modules[i](xyz, features)

        h = features.squeeze(-1)

        return self.mlp(h)
    
class MLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), 0.02)
        x = F.leaky_relu(self.l2(x), 0.02)
        return self.l3(x)

class DMLP(nn.Module):
    def __init__(self, ind, hdim1, hdim2, odim, DP):
        super(DMLP, self).__init__()
        
        self.l1 = nn.Linear(ind, hdim1)
        self.l2 = nn.Linear(hdim1, hdim2)
        self.l3 = nn.Linear(hdim2, odim)
        self.d1 = nn.Dropout(p=DP)
        self.d2 = nn.Dropout(p=DP)
                
    def forward(self, x):
        x = self.d1(F.relu(self.l1(x)))
        x = self.d2(F.relu(self.l2(x)))
        return self.l3(x)
    
class GatedGCN(nn.Module):
    def __init__(
        self,
        num_classes, hidden_dim, node_dim, edge_dim,
        dropout, batch_norm, device,
        per_node_pred=True, sem_num=0, sem_dim = 0
            
    ):
        super(GatedGCN, self).__init__()

        params = {
            'hidden_dim': hidden_dim,            
            'L': 4, 
            'residual': True,
            
            'in_dim': node_dim,
            'in_dim_edge': edge_dim,        
            'n_classes': num_classes,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'device': device,
            'per_node_pred': per_node_pred
        }

        from nets.gated_gcn_net import GatedGCNNet
        
        self.gcn = GatedGCNNet(
            params
        )

        if sem_num > 0:
            self.sem_emb = nn.Embedding(sem_num, sem_dim)
        else:
            self.sem_emb = None

            
    def forward(self, batch_graphs, batch_nf, batch_ef):

        if self.sem_emb is not None:
            cl = batch_graphs.ndata['cl']
            cemb = self.sem_emb(cl)

            node_feat = torch.cat((batch_nf, cemb),dim=1)

        else:
            node_feat = batch_nf
        
        return self.gcn(batch_graphs, node_feat, batch_ef)

class LHSSNet(nn.Module):
    def __init__(
        self, num_classes
    ):
        super(LHSSNet, self).__init__()


        ARCH = {
            'curv': [24,32,64,64],
            'locpca':[36,64,64,64],
            'locvar':[18,32,64,64],
            'spinfea':[192,128,128,128],
            'sc':[144,128,128,128],
            'dhist':[8,32,64,64],
            'pts':[3,16,32,64],
            'ptn':[3,16,32,64]
        }

        
        modules = {}

        for K, L in ARCH.items():
            M = []
            for i in range(0, len(L)-1):
                M += [
                    nn.Linear(L[i], L[i+1]),
                    nn.BatchNorm1d(L[i+1]),
                    nn.ReLU()                    
                ]
            modules[K] = M
        
        self.ll_curv = nn.Sequential(*modules['curv'])
        self.ll_locpca = nn.Sequential(*modules['locpca'])
        self.ll_locvar = nn.Sequential(*modules['locvar'])
        self.ll_spinfea = nn.Sequential(*modules['spinfea'])
        self.ll_sc = nn.Sequential(*modules['sc'])
        self.ll_dhist = nn.Sequential(*modules['dhist'])
        self.ll_pts = nn.Sequential(*modules['pts'])
        self.ll_ptn = nn.Sequential(*modules['ptn'])

        self.final = nn.Sequential(
            nn.Linear(640, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

        self.num_classes = num_classes
        
    def forward(self, _curv, _locpca, _locvar, _spinfea, _sc, _dhist, _pts, _ptn):

        batch_size = _curv.shape[0]
        
        curv = _curv.view(-1, 24)
        locpca = _locpca.view(-1, 36)
        locvar = _locvar.view(-1,18)
        spinfea = _spinfea.view(-1, 192)
        sc = _sc.view(-1, 144)
        dhist = _dhist.view(-1, 8)
        pts = _pts.view(-1, 3)
        ptn = _ptn.view(-1, 3)        
        
        e_curv = self.ll_curv(curv)
        e_locpca = self.ll_locpca(locpca)
        e_locvar = self.ll_locvar(locvar)
        e_spinfea = self.ll_spinfea(spinfea)
        e_sc = self.ll_sc(sc)
        e_dhist = self.ll_dhist(dhist)
        e_pts = self.ll_pts(pts)
        e_ptn = self.ll_ptn(ptn)

        comb = torch.cat((
            e_curv,
            e_locpca,
            e_locvar,
            e_spinfea,
            e_sc,
            e_dhist,
            e_pts,
            e_ptn
        ),dim=1)

        _out = self.final(comb)

        out = _out.view(batch_size, -1, self.num_classes)
        
        return out
        
if __name__ == '__main__':

    device = torch.device('cuda')
    a = torch.randn(7,10000,4).to(device)
    net = PointNetPPCls(10, 1).to(device)
    enc = net(a)
    print(enc.shape)
