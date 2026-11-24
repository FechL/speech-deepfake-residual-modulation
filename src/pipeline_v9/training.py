"""Fixed v9 presets; selection and operating points from train/dev only."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .evaluation import eer
from .config import resolve_families


def make_model(kind, cfg, seed):
    params = dict(cfg['model_params'].get(kind, {}))
    jobs = cfg['n_jobs']
    if kind == 'lr':
        clf = LogisticRegression(class_weight='balanced', random_state=seed, **params)
    elif kind == 'svm_rbf':
        clf = SVC(kernel='rbf', class_weight='balanced', random_state=seed, **params)
    elif kind == 'rf':
        clf = RandomForestClassifier(class_weight='balanced', random_state=seed, n_jobs=jobs, **params)
    elif kind == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return Pipeline([('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=seed, **params))])
    elif kind == 'xgb':
        from xgboost import XGBClassifier
        clf = XGBClassifier(random_state=seed, n_jobs=jobs, eval_metric='logloss', **params)
    elif kind == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(**params)
    elif kind == 'mlp':
        from sklearn.neural_network import MLPClassifier
        if 'hidden_layer_sizes' in params:
            params['hidden_layer_sizes'] = tuple(params['hidden_layer_sizes'])
        clf = MLPClassifier(random_state=seed, **params)
    else:
        raise ValueError(f'Model tidak dikenal: {kind}')
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True,
                                               add_indicator=cfg['missing_indicators'])),
                     ('scaler', StandardScaler()), ('clf', clf)])


def score(spec, frame):
    x = frame[spec['columns']]
    model = spec['model']
    if hasattr(model, 'decision_function'):
        return np.asarray(model.decision_function(x), dtype=float)
    index = list(model.classes_).index(1)
    return model.predict_proba(x)[:, index]


def fit_systems(train, dev, cfg, seed):
    if set(train.speaker_id) & set(dev.speaker_id):
        raise ValueError('Speaker leakage train/dev.')
    families = resolve_families(cfg)
    systems, rows = {}, []
    for family in cfg['families']:
        columns = families[family]
        if train[columns].notna().sum().sum() == 0:
            raise ValueError(f'Semua pengukuran {family} gagal pada train.')
        for kind in cfg['models']:
            model = make_model(kind, cfg, seed)
            model.fit(train[columns], train.label)
            spec = dict(model=model, columns=columns, family=family, classifier=kind)
            scores = score(spec, dev)
            dev_eer, threshold = eer(dev.label, scores)
            spec.update(threshold=threshold, dev_auc=float(roc_auc_score(dev.label, scores)), dev_eer=dev_eer)
            key = f'{family}__{kind}'
            systems[key] = spec
            rows.append(dict(system=key, seed=seed, family=family, classifier=kind, threshold=threshold,
                             dev_auc=spec['dev_auc'], dev_eer=dev_eer))
    champions = {}
    for family in cfg['families']:
        keys = [k for k, s in systems.items() if s['family'] == family]
        champions[family] = max(keys, key=lambda k: (systems[k]['dev_auc'], -systems[k]['dev_eer']))
    return systems, champions, pd.DataFrame(rows)


def grouped_cv(train, cfg, seed):
    folds = min(cfg['group_cv_folds'], train.speaker_id.nunique())
    if folds < 2:
        return pd.DataFrame([{'status': 'unsupported_few_speakers'}])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    families = resolve_families(cfg)
    rows = []
    for fold, (a, b) in enumerate(splitter.split(train, train.label, train.speaker_id)):
        tr, va = train.iloc[a], train.iloc[b]
        if tr.label.nunique() != 2 or va.label.nunique() != 2:
            rows.append(dict(fold=fold, status='unsupported_single_class'))
            continue
        for family in [cfg['baseline_family'], cfg['evidence_family']]:
            cols = families[family]
            spec = dict(model=make_model(cfg['reference_classifier'], cfg, seed), columns=cols)
            spec['model'].fit(tr[cols], tr.label)
            rows.append(dict(fold=fold, family=family, classifier=cfg['reference_classifier'],
                auc=roc_auc_score(va.label, score(spec, va)), n_train=len(tr), n_validation=len(va), status='ok'))
    return pd.DataFrame(rows)
