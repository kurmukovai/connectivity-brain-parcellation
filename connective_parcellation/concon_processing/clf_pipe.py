from scipy.spatial.distance import squareform
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sys import argv

def get_X_y(con_folder):
#     '/data01/ayagoz/sparse_32_concon_HCP/connectomes/Ensemble_parcellation/HE_level3/100/'
    connectomes = sorted(glob(f'{con_folder}*.npy'))

    subject_ids = []
    edges = []
    for c in connectomes:
        sid = int(c.split('/')[-1].split('.')[0])
        if sid != 142626:
            subject_ids.append(sid)
            adj = np.load(c)
            edges.append(squareform(adj))

    meta = pd.read_csv('/home/kurmukov/connective_parcellation_old/HCP/old_code/unrestricted_hcp_freesurfer.csv')
    meta['1_0_gender'] = meta['Gender'].map({'F':0,'M':1})
    sex = dict(zip(meta['Subject'], meta['1_0_gender']))
    y=[sex.get(s) for s in subject_ids]

    y = np.array(y)
    X = np.array(edges)
    return X, y

def model_eval(X, y, random_state=10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25,
                                                        random_state=random_state)
    model = make_pipeline(StandardScaler(), LogisticRegression(penalty = 'l1',
                                                               random_state=random_state,
                                                               solver='liblinear'))
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=random_state)
    score = make_scorer(roc_auc_score, needs_proba=True)
    params={
        
    'logisticregression__C': [0.001, 0.003, 0.01, 0.03,
                              0.1,   0.3,   1,    5,
                              10,    20,    50,   100,
                              150,   180,   200,  220,
                              250,   300,   400,  500]
    }
    gs = GridSearchCV(model,
                      param_grid=params,
                      cv = cv,
                      scoring=score,
                      n_jobs=20,
                      iid=False,
                      refit=True)
    

    gs.fit(X_train, y_train)
    y_pred = gs.predict_proba(X_test)
    auc_test = roc_auc_score(y_test, y_pred[:, 1])
    
    return gs, auc_test

def store_results(gs, auc_test, data_descr=None):
    '''
    data_descr = {
        'ensemble': 'HE',
        'level': 3,
        'sparsity': 10
    }
    '''
    ind = np.argmax(gs.cv_results_['mean_test_score'])
    cv_mean = gs.cv_results_['mean_test_score'][ind]
    cv_std = gs.cv_results_['std_test_score'][ind]
    cv_C = gs.best_params_['logisticregression__C']
    df_results = pd.DataFrame(data = [[cv_mean, cv_std, 
                                      auc_test, cv_C,
                                      data_descr['ensemble'], data_descr['level'],
                                      data_descr['sparsity'], data_descr['n_features'],
                                      data_descr['non_zero']]],
                              columns=['cv_mean', 'cv_std',
                                       'test_mean', 'C',
                                       'ensemble', 'level',
                                       'sparsity', 'n_features',
                                       'non_zero'])
    name = f"{data_descr['ensemble']}_{data_descr['level']}_{data_descr['sparsity']}"
#     df_results.to_csv(f'./model_eval_results/{name}.csv')
    return df_results

if __name__ == "__main__":
    ensemble = argv[1] # HE, CSPA, Aver, Desikan_aparc, Destrieux_aparc2009
#     level = argv[2] # 1,2,3
    if ensemble == 'Desikan_aparc' or ensemble == 'Destrieux_aparc2009':
        df = []
        for sparsity in tqdm(range(10, 101, 10)):
            subject_folder = f'/data01/ayagoz/sparse_32_concon_HCP/connectomes/{ensemble}_resolution/{sparsity}/'
            X, y = get_X_y(subject_folder)
            gs, auc = model_eval(X, y)
            descr = {
                'ensemble': ensemble,
                'level': 0,
                'sparsity': sparsity,
                'n_features': gs.best_estimator_.steps[1][1].coef_.shape[1],
                'non_zero': np.nonzero(gs.best_estimator_.steps[1][1].coef_)[1].shape[0]
            }
            df.append(store_results(gs, auc, descr))

        df_all = df[0]
        for d in df[1:]:
            df_all = df_all.append(d, ignore_index=True)
        name = f"{descr['ensemble']}"
        df_all.to_csv(f'./model_eval_results/{name}.csv')
    else:
        for level in tqdm([1,2,3]):
            df = []
            for sparsity in tqdm(range(10, 101, 10)):
                subject_folder = f'/data01/ayagoz/sparse_32_concon_HCP/connectomes/Ensemble_parcellation/{ensemble}_level{level}/{sparsity}/'
                X, y = get_X_y(subject_folder)
                gs, auc = model_eval(X, y)
                descr = {
                    'ensemble': ensemble,
                    'level': level,
                    'sparsity': sparsity,
                    'n_features': gs.best_estimator_.steps[1][1].coef_.shape[1],
                    'non_zero': np.nonzero(gs.best_estimator_.steps[1][1].coef_)[1].shape[0]
                }
                df.append(store_results(gs, auc, descr))

            df_all = df[0]
            for d in df[1:]:
                df_all = df_all.append(d, ignore_index=True)
            name = f"{descr['ensemble']}_{descr['level']}"
            df_all.to_csv(f'./model_eval_results/{name}.csv')