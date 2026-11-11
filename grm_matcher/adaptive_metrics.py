"""Dataset-level risk ranking and 2D Gaussian calibration (FP32, image units)."""
import math
import torch


def risk_metrics(score, truth):
    score, truth = score.detach().float().cpu(), truth.detach().bool().cpu()
    n = score.numel()
    if n == 0:
        return dict(count=0, auroc=None, auprc=None, accuracy=None, risk_coverage=[])
    order = score.argsort(descending=True, stable=True)
    y = truth[order].float()
    tp, fp = y.cumsum(0), (1-y).cumsum(0)
    # Evaluate ties as one threshold, rather than artificially ordering labels.
    ends = torch.cat((score[order][1:] != score[order][:-1], torch.ones(1, dtype=torch.bool)))
    precision = tp[ends] / (tp[ends]+fp[ends])
    positives, negatives = int(truth.sum()), int((~truth).sum())
    auprc, auroc = None, None
    if positives:
        recall = tp[ends]/positives
        auprc = float(((recall-torch.cat((torch.zeros(1), recall[:-1]))) * precision).sum())
    if positives and negatives:
        tpr = torch.cat((torch.zeros(1), tp[ends]/positives))
        fpr = torch.cat((torch.zeros(1), fp[ends]/negatives))
        auroc = float(torch.trapezoid(tpr, fpr))
    low = truth[score.argsort(stable=True)].float().cumsum(0)
    curve = []
    for fraction in (.05, .1, .2, .3, .5, .7, .9, 1.):
        k = max(1, int(n*fraction))
        curve.append(dict(coverage=k/n, expansion_need_rate=float(low[k-1]/k)))
    return dict(count=n, positives=positives, auroc=auroc, auprc=auprc,
                accuracy=float(((score >= .5) == truth).float().mean()), risk_coverage=curve)


def covariance_metrics(covariance, error, floor=1e-6):
    cov, e = covariance.detach().float().cpu(), error.detach().float().cpu()
    n = len(e)
    if not n:
        return dict(count=0, mahalanobis_nll=None, coverage68=None, coverage95=None,
                    trace_squared_error_correlation=None, calibration_curve=[])
    # Base spatial covariance can be singular; report the numerical jitter used.
    cov = (cov+cov.transpose(-1, -2))*.5 + floor*torch.eye(2)
    chol = torch.linalg.cholesky(cov)
    mahal = (e[..., None]*torch.cholesky_solve(e[..., None], chol)).sum((-1, -2))
    logdet = 2*chol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
    error2 = e.square().sum(-1)
    corr = None
    if n > 1 and trace.std() > 0 and error2.std() > 0:
        corr = float(torch.corrcoef(torch.stack((trace, error2)))[0, 1])
    levels = (.1, .3, .5, .68, .8, .9, .95, .99)
    curve = [dict(nominal=p, observed=float((mahal <= -2*math.log(1-p)).float().mean())) for p in levels]
    return dict(count=n, mahalanobis_nll=float((.5*(mahal+logdet)+math.log(2*math.pi)).mean()),
                coverage68=curve[3]['observed'], coverage95=curve[6]['observed'],
                trace_squared_error_correlation=corr, calibration_curve=curve,
                covariance_calibration_error=sum(abs(row['nominal']-row['observed']) for row in curve)/len(curve),
                numerical_jitter_pixel_squared=floor)
