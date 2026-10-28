"""Prospective calibration with per-domain source-group support and uncertainty.

Controls apply to represented calibration domains, not arbitrary unseen domains.
"""
from collections import defaultdict
import math


def binomial_upper(k, n, alpha):
    """One-sided Clopper-Pearson upper bound, obtained by binomial inversion."""
    if n <= 0 or not 0 < alpha < 1 or not 0 <= k <= n:
        raise ValueError('Invalid binomial interval input')
    if k == n:
        return 1.
    lo, hi = k / n, 1.
    for _ in range(65):
        p = (lo + hi) / 2
        logs = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)
                + i*math.log(p) + (n-i)*math.log1p(-p) for i in range(k+1)]
        top = max(logs)
        cdf = math.exp(top) * sum(math.exp(v-top) for v in logs)
        if cdf > alpha:
            lo = p
        else:
            hi = p
    return (lo + hi) / 2


def domain_metrics(rows, threshold, alpha):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['label'] not in (0, 1) or not math.isfinite(row['risk']) or not 0 <= row['risk'] <= 1:
            raise ValueError('Invalid calibration label or score')
        groups[row['domain']][row['source_group'], row['label']].append(row['risk'] >= threshold)
    output = {}
    for domain, values in groups.items():
        benign = [any(v) for (g, label), v in values.items() if label == 0]
        attack = [sum(v)/len(v) for (g, label), v in values.items() if label == 1]
        fp, n = sum(benign), len(benign)
        output[domain] = {'benign_source_groups': n, 'benign_groups_rejected': fp,
                          'fpr': fp/n if n else None,
                          'fpr_upper': binomial_upper(fp, n, alpha) if n else None,
                          'attack_source_groups': len(attack),
                          'source_balanced_recall': sum(attack)/len(attack) if attack else None}
    return output


def select_threshold(rows, required_domains, fpr_limit=.05, minimum_recall=.5,
                     minimum_benign_groups=100, family_alpha=.05):
    if not rows or not required_domains or len(set(required_domains)) != len(required_domains):
        raise ValueError('Nonempty rows and distinct required domains are required')
    if {r['domain'] for r in rows} != set(required_domains):
        raise ValueError('Calibration domains differ from protocol')
    alpha = family_alpha / len(required_domains)
    support = domain_metrics(rows, 1., alpha)
    config = {'fpr_limit': fpr_limit, 'minimum_recall': minimum_recall,
              'minimum_benign_groups': minimum_benign_groups, 'family_alpha': family_alpha,
              'per_domain_alpha': alpha, 'required_domains': list(required_domains)}
    if any(v['benign_source_groups'] < minimum_benign_groups or not v['attack_source_groups'] for v in support.values()):
        return {'status': 'insufficient_calibration_support', 'threshold': None, 'domains': support, 'config': config}
    candidates = sorted({r['risk'] for r in rows} | {math.nextafter(max(r['risk'] for r in rows), math.inf)})
    eligible = []
    for threshold in candidates:
        metrics = domain_metrics(rows, threshold, alpha)
        if all(m['fpr_upper'] <= fpr_limit and m['source_balanced_recall'] >= minimum_recall for m in metrics.values()):
            eligible.append((min(m['source_balanced_recall'] for m in metrics.values()),
                             -max(m['fpr_upper'] for m in metrics.values()), threshold, metrics))
    if not eligible:
        return {'status': 'no_useful_operating_point', 'threshold': None, 'config': config}
    _, _, threshold, metrics = max(eligible, key=lambda x: x[:3])
    return {'status': 'selected', 'threshold': threshold, 'domains': metrics, 'config': config}
