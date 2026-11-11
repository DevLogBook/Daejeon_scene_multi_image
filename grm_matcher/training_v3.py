"""V3 supervision: bidirectional coarse labels, anchors, predicted initialization and confidence."""
from dataclasses import replace
import torch
import torch.nn.functional as F
from .architecture_v3 import confidence_logits, matching_probability
from .coordinates import flat_indices_to_image_xy, image_xy_to_normalized_grid
from .stage25 import GRMStage25CoarseOutput, GRMStage25StudentOutput, transform_points
from .imc_dataset import sampson_squared, sparse_bidirectional_labels
from .confidence_supervision import confidence_targets


def agricultural_v3(objective, model, batch):
    if 'gt_dense_mapping_ba' in batch:
        forward = dense_direction_labels(batch, 'a', model.config, objective.adaptation)
        backward = dense_direction_labels(batch, 'b', model.config, objective.adaptation)
        return supervised_v3(objective, model, batch['image_a'], batch['image_b'],
                             forward, backward, batch.get('valid_a'), batch.get('valid_b'), batch=batch)
    from .scratch_transfer import exact_h_labels
    labels = exact_h_labels(batch, model.config, objective.adaptation)
    sparse = dict(image_a=batch['image_a'], sparse_a=labels.src_xy,
        sparse_b=labels.fine_dst_xy, sparse_valid=labels.valid_mask,
        sparse_confidence=labels.confidence)
    bidirectional = sparse_bidirectional_labels(sparse, labels.src_xy.device, model.config, objective.adaptation)
    return supervised_v3(objective, model, batch['image_a'], batch['image_b'],
        bidirectional.forward, bidirectional.backward, batch.get('valid_a'), batch.get('valid_b'), batch=batch)


def dense_direction_labels(batch, side, config, adaptation):
    """Use each direction's true field, not swapped AB points or inverse H."""
    image = batch['image_'+side]
    b, _, h, w = image.shape
    y, x = torch.meshgrid(torch.arange(h//config.coarse_stride, device=image.device),
                          torch.arange(w//config.coarse_stride, device=image.device), indexing='ij')
    source = (torch.stack((x,y),-1).float().reshape(-1,2)+.5)*config.coarse_stride-.5
    indices = torch.linspace(0,len(source)-1,min(len(source),config.max_source_queries),device=image.device).round().long()
    source = source[indices][None].expand(b,-1,-1)
    grid = image_xy_to_normalized_grid(source,h,w).unsqueeze(2)
    field = batch['gt_dense_mapping' if side == 'a' else 'gt_dense_mapping_ba']
    target = F.grid_sample(field.float(),grid,align_corners=False)[:,:, :,0].transpose(1,2)
    visible = F.grid_sample(batch['gt_overlap_'+side].float(),grid,align_corners=False)[:,0,:,0] > .999
    valid = visible & torch.isfinite(target).all(-1)
    valid &= (target[...,0]>=0)&(target[...,0]<=w-1)&(target[...,1]>=0)&(target[...,1]<=h-1)
    sparse = dict(image_a=image,sparse_a=source,sparse_b=target,
                  sparse_valid=valid,sparse_confidence=valid.float())
    return sparse_bidirectional_labels(sparse,image.device,config,adaptation).forward


def supervised_v3(objective, model, a, b, forward, backward, valid_a, valid_b,
                  fundamental=None, batch=None):
    count = int(forward.valid_mask.sum()+backward.valid_mask.sum())
    if not count:
        return None, {'valid_count': 0.}
    config = model.config
    output = None
    if batch is not None:
        if config.adaptive_enabled and config.adaptive_mode == 'oracle_5_9':
            from .adaptive import oracle_forward
            output, aux = oracle_forward(model, batch, return_auxiliary=True)
        else:
            output, aux = model(a, b, return_auxiliary=True, valid_a=valid_a, valid_b=valid_b)
        fa, fb, corr = aux.features_a, aux.features_b, aux.correlation
    else:
        fa, fb, ta, tb = model.encode_pair(a, b, valid_a, valid_b)
        corr = model.correlation(ta, tb, fa['valid_coarse'].flatten(1), fb['valid_coarse'].flatten(1))
    h, w = fa['F_coarse'].shape[-2:]
    coarse = objective.coarse(GRMStage25CoarseOutput(corr.similarity,
        corr.matchability_a, corr.matchability_b, h, w), forward, backward)['total']
    # Known anchors keep both refiners supervised even before coarse matching works.
    am, af = model.refine_points(fa, fb, forward.src_xy, forward.coarse_dst_xy, forward.valid_mask)
    anchor_labels = replace(forward, valid_mask=forward.valid_mask & af.valid_mask)
    anchor = objective.refine(GRMStage25StudentOutput(am, af), anchor_labels)['total']
    cells = torch.round((forward.src_xy+.5)/config.coarse_stride-.5).long()
    inside = (cells[..., 0]>=0)&(cells[..., 0]<w)&(cells[..., 1]>=0)&(cells[..., 1]<h)
    query = cells[..., 1].clamp(0, h-1)*w + cells[..., 0].clamp(0, w-1)
    probability = matching_probability(corr, config)
    target = probability.argmax(-1).gather(1, query)
    initial = flat_indices_to_image_xy(target, w, config.coarse_stride)
    risk_logits, expand, routing_labels = None, None, None
    if config.adaptive_enabled:
        from .adaptive import route_points, search_labels, risk_loss, variance_nll
        routing_valid = forward.valid_mask & inside & model.points_valid(forward.src_xy, fa)
        routing_valid &= model.points_valid(forward.fine_dst_xy, fb)
        risk_logits, expand = route_points(model, corr, query, target, initial, routing_valid,
                                           forward.fine_dst_xy, routing_valid, compute_risk=True)
        routing_labels = search_labels(initial, forward.fine_dst_xy, routing_valid, config.mid_stride)
    pm, pf = model.refine_points(fa, fb, forward.src_xy, initial, forward.valid_mask & inside, mid_expand=expand)
    known = forward.valid_mask & inside & pf.valid_mask
    reachable = ((forward.fine_dst_xy-initial).abs() <= (config.mid_window_size//2)*config.mid_stride).all(-1)
    if config.adaptive_enabled:
        radius = (2 + 2*expand.long()) if config.adaptive_mode in ('oracle_5_9', 'learned_5_9') else int(config.adaptive_mode[-1])//2
        reachable = ((forward.fine_dst_xy-initial).abs() <= (radius[..., None] if torch.is_tensor(radius) else radius)*config.mid_stride).all(-1)
    predicted_labels = replace(forward, coarse_dst_xy=initial, valid_mask=known & reachable)
    predicted = objective.refine(GRMStage25StudentOutput(pm, pf), predicted_labels)['total']
    # Sparse augmented supervision has no exact reverse field at arbitrary endpoints.
    # Use reverse labels only when no dense/H batch is available for Oracle.
    if config.adaptive_enabled and config.adaptive_mode == 'oracle_5_9' and batch is None:
        from .cycle_consistency import GRMCycleConsistencyOutput
        z = torch.zeros_like(known, dtype=torch.float32)
        cycle = GRMCycleConsistencyOutput(forward.src_xy, query, z, z, known & False)
    else:
        cycle = model.direct_cycle(fa, fb, probability, forward.src_xy, pf.dst_xy, known,
                                  correlation=corr, oracle_batch=batch if config.adaptive_enabled and config.adaptive_mode == 'oracle_5_9' else None)
    logits = confidence_logits(model, corr, query, target, pm, pf, cycle)
    error = (pf.dst_xy-forward.fine_dst_xy).norm(dim=-1)
    truth = (error.detach() <= 3).float()
    zero = coarse * 0
    confidence, brier = zero, zero.detach()
    if logits is not None and known.any():
        confidence = F.binary_cross_entropy_with_logits(logits[known], truth[known])
        brier = (logits[known].sigmoid()-truth[known]).square().mean().detach()
    # Only cycle-supervise already GT-consistent forward predictions.
    cycle_mask = known & (error.detach() <= 3) & cycle.valid_mask
    cycle_loss = F.smooth_l1_loss(cycle.returned_src_xy[cycle_mask], forward.src_xy[cycle_mask]) if cycle_mask.any() else zero
    epipolar = zero
    if fundamental is not None and anchor_labels.valid_mask.any():
        distance = sampson_squared(forward.src_xy, af.dst_xy, fundamental)
        epipolar = (torch.sqrt(1+distance[anchor_labels.valid_mask])-1).mean()
    overlap = zero
    metrics = {}
    if batch is not None:
        # Exact synthetic visibility teaches negative matchability (including Flow).
        terms = []
        for suffix, probability, feature in (('a', corr.matchability_a, fa), ('b', corr.matchability_b, fb)):
            truth_map = F.interpolate(batch['gt_overlap_'+suffix].float(), (h,w), mode='nearest-exact').flatten(1)
            mask = feature['valid_coarse'].flatten(1)
            if mask.any():
                terms.append(F.binary_cross_entropy(probability[mask].clamp(1e-6, 1-1e-6), truth_map[mask]))
        overlap = sum(terms)/max(1,len(terms))
        metrics.update(predicted_matches=0., correct_at_1px=0., correct_at_3px=0., correct_at_5px=0.)
    if batch is not None and output.num_matches > 0:
        grid = image_xy_to_normalized_grid(output.src_xy, a.shape[-2], a.shape[-1]).unsqueeze(2)
        visible = F.grid_sample(batch['gt_overlap_a'].float(), grid, mode='nearest', align_corners=False)[:,0,:,0] > .5
        if 'gt_dense_mapping_ba' in batch:
            visible = F.grid_sample(batch['gt_overlap_a'].float(),grid,align_corners=False)[:,0,:,0] > .999
            actual_gt = F.grid_sample(batch['gt_dense_mapping'].float(),grid,align_corners=False)[:,:,:,0].transpose(1,2)
        else:
            actual_gt = transform_points(output.src_xy, batch['H_ab'])
        actual_error = (output.dst_xy-actual_gt).norm(dim=-1)
        label_known = None
        if 'gt_confidence_known_a' in batch:
            label_known = F.grid_sample(batch['gt_confidence_known_a'].float(),grid,
                align_corners=False)[:,0,:,0] > .999
        actual_truth, actual_valid, _ = confidence_targets(visible, output.valid_mask, actual_error,
            torch.ones_like(actual_error), real=False, label_known=label_known)
        if aux.confidence_logits is not None and actual_valid.any():
            actual_conf = F.binary_cross_entropy_with_logits(aux.confidence_logits[actual_valid], actual_truth[actual_valid])
            confidence = .5*(confidence+actual_conf)
            brier = (output.p_match[actual_valid]-actual_truth[actual_valid]).square().mean().detach()
        metrics.update(predicted_matches=float(output.valid_mask.sum()),
            correct_at_3px=float((actual_truth.bool() & output.valid_mask).sum()),
            correct_at_1px=float((visible & (actual_error<=1) & output.valid_mask).sum()),
            correct_at_5px=float((visible & (actual_error<=5) & output.valid_mask).sum()),
            output_confidence_supervised_count=float(actual_valid.sum()),
            output_confidence_negative_count=float((actual_valid & ~actual_truth.bool()).sum()),
            output_confidence_ignored_count=float((output.valid_mask & ~actual_valid).sum()))
    total = coarse + objective.anchor_weight*anchor + predicted + config.v3_confidence_weight*confidence + .05*cycle_loss + .1*overlap + objective.epipolar_weight*epipolar
    if config.adaptive_enabled:
        # Train on the same selected queries and centers as image-only deployment.
        # Fine failure must not erase a difficult coarse query's risk supervision.
        risk_labels_for_loss, risk_logits_for_loss = routing_labels, risk_logits
        if batch is not None:
            from .adaptive import ground_truth_at, risk_features
            selected_truth, selected_visible = ground_truth_at(batch, output.src_xy)
            selected_initial = flat_indices_to_image_xy(aux.target_indices, w, config.coarse_stride)
            risk_labels_for_loss = search_labels(selected_initial, selected_truth,
                selected_visible & aux.query_valid_mask, config.mid_stride)
            risk_logits_for_loss = aux.risk_logits
            if risk_logits_for_loss is None:
                features = risk_features(corr, aux.query_indices, aux.target_indices, config.temperature)
                risk_logits_for_loss = model.search_risk_head(features.detach() if config.adaptive_detach_risk else features)
        risk = risk_loss(risk_logits_for_loss, risk_labels_for_loss, config.adaptive_risk_focal_gamma) if risk_logits_for_loss is not None else total*0
        var = total*0
        if config.adaptive_variance:
            if batch is not None and aux.calibrated_covariance is not None:
                var = variance_nll(aux.calibrated_covariance, output.dst_xy-selected_truth,
                                   selected_visible & output.valid_mask)
            else:
                covariance = model.variance_calibrator(pf)
                var = variance_nll(covariance, pf.dst_xy-forward.fine_dst_xy, known)
        total = total + config.adaptive_risk_weight*risk + config.adaptive_variance_weight*var
        metrics.update(risk_loss=risk.detach(), variance_loss=var.detach(),
                       risk_supervised_count=float(risk_labels_for_loss.supervised.sum()),
                       risk_positive_count=float(risk_labels_for_loss.expand.sum()),
                       risk_query_count=float(risk_labels_for_loss.category.numel()),
                       risk_invisible_count=float((risk_labels_for_loss.category == 3).sum()),
                       expanded_count=float(aux.expanded_mask.sum()) if batch is not None and aux.expanded_mask is not None else float(expand.sum()),
                       unrecoverable_count=float((risk_labels_for_loss.category == 2).sum()))
    metrics.update(loss=total.detach(), coarse=coarse.detach(), anchor=anchor.detach(),
        refinement=predicted.detach(), confidence=confidence.detach(), confidence_brier=brier,
        cycle=cycle_loss.detach(), overlap=overlap.detach(), epipolar=epipolar.detach(),
        valid_count=float(count), confidence_supervised_count=float(known.sum()),
        anchor_predicted_correct_3px=float((truth.bool() & known).sum()))
    if batch is not None and 'gt_dense_mapping_ba' in batch:
        supervised = forward.valid_mask & inside
        metrics.update(augmentation_strength=float(batch['augmentation_strength'].float().mean()),
                       flow_fraction=float(batch['flow_active'].float().mean()),
                       reachable_fraction=float((reachable & supervised).sum()/supervised.sum().clamp_min(1)),
                       coarse_initial_error_mean=float((initial-forward.fine_dst_xy).norm(dim=-1)[supervised].mean()) if supervised.any() else 0.,
                       inverse_residual_max=float(batch['inverse_residual_max'].max()))
        if 'augmentation_tier' in batch:
            for tier,name in enumerate(('unchanged','weak','medium','strong','severe')):
                metrics['tier_'+name+'_fraction']=float((batch['augmentation_tier']==tier).float().mean())
    return total, metrics
