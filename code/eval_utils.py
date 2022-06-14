import torch
import utils
import time
import data_utils
from copy import deepcopy, copy
import numpy as np
import heapq
from tqdm import tqdm
import os

def search_beam(probs, num, keep_ll=False):
    with torch.no_grad():
        return _search_beam(probs, num, keep_ll)

def _search_beam(probs, num, keep_ll):
    LL = torch.log(probs[0]+1e-8).cpu()
    S = [[i] for i in range(probs[0].shape[0])]

    for i in range(1, len(probs)):
        LLi = torch.log(probs[i] +1e-8).cpu()

        R = (LL.view(-1, 1) + LLi.view(1, -1)).flatten()

        Rv, Ri = torch.topk(R, min(num, R.shape[0]))

        I0, I1 = np.unravel_index(Ri.cpu().numpy(), (LL.shape[0], LLi.shape[0]))

        nS = []
        nLL = []

        LL = LL[torch.from_numpy(I0)] + LLi[torch.from_numpy(I1)]
        
        for i0, i1 in zip(I0, I1):
            nS.append(S[i0] + [i1])

        S = nS

    if keep_ll:
        return [(ll.item(), torch.tensor(s)) for ll, s in zip(LL,S)]              
    else:                
        return [torch.tensor(s) for s in S]

                
def calc_eval_segment_metrics(results, grammar):

    tars = []
    best_seg_accs = []
    first_seg_accs = []

    best_pt_accs = []
    first_pt_accs = []
    
    best_preds = []
    first_preds = []
    labels = []
    
    for samples, targets, pt_segments, pt_labels in results:
                
        labels.append(pt_labels)
                        
        samples = torch.stack(samples,dim=0)
        seg_accs = (targets.view(1,-1) == samples).float().mean(dim=1)

        first_seg_accs.append(seg_accs[0].item())
        best_seg_accs.append(seg_accs.max().item())

        if seg_accs.max().item() == 1.0:
            tars.append(1.)
        else:
            tars.append(0.)
        
        bind = seg_accs.argmax().item()
                                                                            
        best_pred = torch.zeros(pt_labels.shape).long() - 1
        first_pred = torch.zeros(pt_labels.shape).long() - 1

        best_seg_pred = samples[bind]
        first_seg_pred = samples[0]
        
        for i in np.unique(pt_segments):
            sinds = (pt_segments == i).nonzero().squeeze()

            best_pred[sinds] = best_seg_pred[i]
            first_pred[sinds] = first_seg_pred[i]

        assert (best_pred >= 0).all() and (first_pred >= 0).all(), 'somethign went wrong'

        best_preds.append(best_pred)
        first_preds.append(first_pred)

        best_pt_accs.append((best_pred == pt_labels).float().mean().item())
        first_pt_accs.append((first_pred == pt_labels).float().mean().item())
        
        
    best_iou = calc_mIoU(labels, best_preds, grammar)
    first_iou = calc_mIoU(labels, first_preds, grammar)
        
    return {
        'seg_tar': torch.tensor(tars).mean().item(),
        'seg_best_acc': torch.tensor(best_seg_accs).mean().item(),
        'seg_first_acc': torch.tensor(first_seg_accs).mean().item(),
        'pt_best_acc': torch.tensor(best_pt_accs).mean().item(),
        'pt_first_acc': torch.tensor(first_pt_accs).mean().item(),
        'best_iou': best_iou,
        'first_iou': first_iou
    }
        
def calc_Acc(labels, preds):
    corr = 1e-8
    total = 1e-8

    for l, p in zip(labels, preds):
        corr += (l == p).sum().item()
        total += l.shape[0] * 1.

    return corr / total

def calc_mIoU(labels, preds, grammar):
    ious = []
    for node in grammar.node_map:
    
        ninds = grammar.node_map[node][0]
        ntype = grammar.node_map[node][1]

        inter = 0
        union = 0
        
        for pred, label in zip(preds, labels):

            pn = torch.zeros(pred.shape).int()
            ln = torch.zeros(label.shape).int()

            for ind in ninds:                
                
                pind = (pred == ind).nonzero().squeeze()
                lind = (label == ind).nonzero().squeeze()
               
                pn[pind] = 1
                ln[lind] = 1

            inter += (pn & ln).sum().item()
            union += (pn | ln).sum().item()            
    
        if union == 0:
            continue
            
        iou = (1.*inter) / (1.*union)

        ious.append(iou)

    miou = torch.tensor(ious).float().mean().item()
    return miou


def calc_eval_point_metrics(results, grammar):

    all_ious = []
        
    for node in grammar.node_map:
    
        ninds = grammar.node_map[node][0]
        ntype = grammar.node_map[node][1]

        inter = 0
        union = 0
        
        for pred, label in results:

            pn = torch.zeros(pred.shape).int()
            ln = torch.zeros(label.shape).int()

            for ind in ninds:                
                
                pind = (pred == ind).nonzero().squeeze()
                lind = (label == ind).nonzero().squeeze()
               
                pn[pind] = 1
                ln[lind] = 1

            inter += (pn & ln).sum().item()
            union += (pn | ln).sum().item()            
    
        if union == 0:
            continue
            
        iou = (1.*inter) / (1.*union)

        all_ious.append(iou)
    
    all_iou = torch.tensor(all_ious).float().mean().item()
    
    return {
        'all_iou': all_iou,
    }
    
def model_eval_samples(
    args,
    net,
    loader,
    log_info,
    grammar,
    eval_fn,
):
    sample_results = []
    
    for count, (_points, _labels, _segments, _seg_labels, inds) in tqdm(enumerate(loader), total=loader.num_to_eval):

        points = _points[0]
        labels = _labels[0]
        seg_labels = _seg_labels[0]
        segments = _segments[0]

        eval_inp = {
            'net': net,
            'points': points,
            'segments': segments,
            'num_segments': seg_labels.shape[0],
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'mode': 'sample',
            'keep_ll': False
        }
                
        samples = eval_fn(
            eval_inp
        )

        sample_results.append((samples, seg_labels, segments, labels))        
                            
    utils.log_print("Calculating Metrics", args)
        
    results = calc_eval_segment_metrics(
        sample_results, grammar
    )
    results['nc'] = 1.

    utils.log_print("Sample Metric Values:", args)

    utils.print_results(
        log_info,
        results,
        args
    )
    
            
def model_eval(
    args,
    loader,
    net,
    e,
    grammar,
    name,
    model_eval_fn,
):
        
    point_results = []

    corr = 0
    total = 0

    total_ll = 0.
    
    for count, (_points, _labels, _segments, _seg_labels, inds) in enumerate(loader):

        points = _points[0]
        labels = _labels[0]
        seg_labels = _seg_labels[0]
        segments = _segments[0]

        eval_inp = {
            'net': net,
            'points': points,
            'segments': segments,
            'num_segments': seg_labels.shape[0],
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'mode': 'point',
            'ind': inds[0]
        }
        
        # preds: tensor where each point is given a terminal 
        preds, seg_dist = model_eval_fn(eval_inp)

        ll = 0.
        
        if seg_dist is not None:
            for i,sl in enumerate(seg_labels):
                ll += torch.log(seg_dist[i][sl] + 1e-40).item()

        total_ll += ll
                
        point_results.append((preds, labels))

        _corr = (preds == labels).sum().item()
        _total = labels.shape[0]    

        corr += _corr
        total += _total
        
        ind = inds[0]
        
    metric_result = calc_eval_point_metrics(point_results, grammar)

    metric_result['ll'] = total_ll
    metric_result['corr'] = corr
    metric_result['total'] = total
    
    metric_result['count'] = count
    metric_result['nc'] = 1
    
    return metric_result


def calc_search_eval_metrics(all_preds):
    res = {}

    itotal = 0.
    icorr = 0.

    fp = 1e-8
    fn = 1e-8
    tp = 1e-8
    tn = 1e-8

    prob_margin = []
    pos_first = []

    pos_better = []
    
    for preds in all_preds:
        tp += 1.
        itotal += preds.shape[0] * 1.
                
        if preds[0] > 0.5:
            icorr += 1.
        else:
            fn += 1.

        neg_corr = (preds[1:] < 0.5).sum().item()
        num_neg = (preds.shape[0] * 1.) - 1

        icorr += neg_corr
        fp += num_neg - neg_corr
        tn += num_neg

        if preds.shape[0] == 1:
            margin = 0.
        else:
            margin = preds[0].item() - preds[1:].max().item()

        prob_margin.append(margin)
        pos_first.append(1. if margin >= 0 else 0.)

        pos_better.append(
            (preds[1:] < preds[0]).sum().item() / ((preds[1:].shape[0] * 1.) + 1e-8)
         )
        
    pos_acc = (tp-fn) / tp
    neg_acc = (tn-fp) / tn
    mean_acc = (pos_acc + neg_acc) / 2.

    fpr = fp / tn
    fnr = fn / tp

    wm = ((3. * (1-fnr)) + (1-fpr)) / 4
    
    res['acc'] = icorr / itotal
    res['mean_acc'] = mean_acc
    res['fp'] = fpr
    res['fn'] = fnr
    res['wm'] = wm
    res['prob_margin'] = torch.tensor(prob_margin).float().mean().item()
    res['pos_first'] = torch.tensor(pos_first).float().mean().item()
    res['pos_better'] = torch.tensor(pos_better).float().mean().item()

    return res
    
def search_model_eval(
    args,
    loader,
    net,
    e,
    grammar,
    name,
    model_eval_fn,
):
            
    # list of N x K preds
    # N shapes, K-1 negative examples per shape, first index always positive
    preds = []
    losses = []
    
    for count, (meshes, samps, segments, seg_map, pos_info, neg_info, ind) in enumerate(loader):
        
        eval_inp = {
            'net': net,
            'meshes': meshes,
            'samps': samps,
            'segments': segments,
            'seg_map': seg_map,
            'pos_info': pos_info,
            'neg_info': neg_info,
            'grammar': grammar,
            'epoch': e,
            'ind': ind,
            'name': name,
            'batch_size': args.batch_size,
            'args': args
        }
                
        pred, loss = model_eval_fn(eval_inp)
        preds.append(pred)
        losses.append(loss)
        
    metric_result = calc_search_eval_metrics(preds)
    metric_result['neg_loss'] = -1 * torch.tensor(losses).mean().item()
    metric_result['nc'] = 1
    
    return metric_result


def run_eval_epoch(
    args,
    res,
    net,
    eval_data,
    EVAL_LOG_INFO,
    e,
    grammar,
    model_eval_fn,
    sub_fn = model_eval
):
        
    if (e+1) % args.eval_per != 0:
        return -1
        
    with torch.no_grad():
        
        if isinstance(net, dict):
            for n in net.values():
                n.eval()
        else:
            net.eval()        

        
            
        t = time.time()                

        eval_results = {}
        for key, loader in eval_data:

            eval_results[key] = sub_fn(
                args,
                loader,
                net,
                e,
                grammar,
                key,
                model_eval_fn
            )
            
            utils.log_print(
                f"Evaluation {key} set results:",
                args
            )

            utils.print_results(
                EVAL_LOG_INFO,
                eval_results[key],
                args
            )
                        
        utils.log_print(f"Eval Time = {time.time() - t}", args)

        res['eval_epochs'].append(e)
                
        utils.make_plots(
            EVAL_LOG_INFO,
            eval_results,            
            res['eval_plots'],
            res['eval_epochs'],
            args,
            'eval'
        )

    if 'val' not in res['eval_plots']:
        utils.log_print("!!!MISSING EVAL ASSUMING TESTING!!!!", args)
        return -1
    
    eps = res['eval_epochs']
    metric_res = torch.tensor(res['eval_plots']['val'][args.es_metric])
    cur_ep = eps[-1]
    
    for i, ep in enumerate(eps):
        if cur_ep - ep <= args.es_patience:
            metric_res[i] -= args.es_threshold

    best_ep_ind = metric_res.argmax().item()
    best_ep = eps[best_ep_ind]
    
    if cur_ep - best_ep > args.es_patience:
        utils.log_print(
            f"Stopping early at epoch {cur_ep}, "
            f"choosing epoch {best_ep} with val {args.es_metric} " 
            f"of {metric_res.max().item()}",
            args
        )
        utils.log_print(
            f"Final test value for {args.es_metric} : {res['eval_plots']['test'][args.es_metric][best_ep_ind]}",
            args
        )
        return best_ep

    return -1
        
