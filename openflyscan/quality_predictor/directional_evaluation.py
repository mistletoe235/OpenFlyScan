"""Fixed region budgets; view observations never become independent selection slots."""

import numpy as np
from scipy.stats import spearmanr

from .region_selection import select_regions


def regional_metrics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    result = dict(regions=len(target), recall10=None, recall20=None, rho=None)
    if not len(target):
        return result
    for fraction, name in [(.1, 'recall10'), (.2, 'recall20')]:
        budget = int(np.ceil(fraction*len(target)))
        chosen = np.argsort(-prediction, kind='stable')[:budget]
        truth = np.argsort(-target, kind='stable')[:budget]
        result[name] = len(set(chosen) & set(truth))/budget
    if len(target) > 1 and np.std(prediction) > 1e-8 and np.std(target) > 1e-8:
        result['rho'] = float(spearmanr(prediction, target).statistic)
    return result


def scene_region_evaluation(rows):
    scenes = {}
    for row in rows:
        if row['state'] != 'full':
            continue
        for index, evidence in enumerate(row['region_evidence']):
            scenes.setdefault(row['scene'], []).append({**evidence, 'scene': row['scene'],
                'chunk': str(row['seed']), 'score': 0., 'prediction': row['prediction'][index],
                'target': row['target'][index]})
    results = {}
    for scene, candidates in scenes.items():
        clusters = select_regions(candidates, len(candidates))['merged_regions']
        prediction = [max(candidates[index]['prediction'] for index in group['members']) for group in clusters]
        target = [max(candidates[index]['target'] for index in group['members']) for group in clusters]
        results[scene] = {**regional_metrics(prediction, target), 'input_regions': len(candidates),
                          'duplicates_removed': len(candidates)-len(clusters)}
    return dict(per_scene=results,
        macro_recall10=float(np.mean([value['recall10'] for value in results.values() if value['recall10'] is not None]))
            if any(value['recall10'] is not None for value in results.values()) else None,
        macro_recall20=float(np.mean([value['recall20'] for value in results.values() if value['recall20'] is not None]))
            if any(value['recall20'] is not None for value in results.values()) else None,
        grouping='Score/label-blind conservative shared-image matching; unknown matches remain separate.',
        budget='ceil(fraction * deduplicated eligible regions), independently per scene; never per angle.',
        boundary='Sampled 32 chunks, not exhaustive map coverage; no approved outer ROI. Not comparable to old chunk-median recall.')
