import sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

# Data processing

def get_scores(anime):
    '''
    Cleanup/preprocess scores values.
    '''
    
    scored_df = (
        anime[['anime_id', 'score', 'soup']]
        .query('score > 0')
    )
    
    return scored_df

## Training

def leave_last_out(data, key):
    sorted = data.sort_values(['profile_idx', key])
    holdout = sorted.drop_duplicates(subset=['profile_idx'], keep='last').reset_index(drop=True)
    train = sorted.drop(holdout.index).reset_index(drop=True)
    return train, holdout

def get_user_profiles(item_training, training, holdout):
    
    sum_strengthes = 0.
    profiles = []
    
    for idx in holdout.profile_idx:
        profile_idxs = training[training.profile_idx == idx].index
        
        user_weights = training.iloc[profile_idxs][['score']].values
        
        user = np.sum(item_training[profile_idxs] * user_weights, axis=0)
        
        profiles.append(user)
        sum_strengthes += np.sum(user_weights)
    
    user_item_strengths_weighted_avg = np.array(profiles) / sum_strengthes
    
    return user_item_strengths_weighted_avg

## Evaluation

def simple_model_recom_func(scores, topn=10):
    recommendations = np.apply_along_axis(topidx, 1, scores, topn)
    return recommendations

def topidx(a, topn):
    parted = np.argpartition(a, -topn)[-topn:]
    return parted[np.argsort(-a[parted])]

def model_evaluate(recommended_items, holdout, data, topn=10):
    holdout_items = holdout.anime_id.values
    assert recommended_items.shape[0] == len(holdout)
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    
    # HR calculation
    hr = np.mean(hits_mask.any(axis=1))
    
    # MRR calculation
    n_test_users = recommended_items.shape[0]
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / len(hits_mask)
    
    # coverage calculation
    coverage = len(np.unique(recommended_items)) / data.anime_id.nunique()
    
    return hr, mrr, coverage

## Random recommendataions

def build_random_model(config, trainset):
    n_items = trainset[trainset.anime_id.name].max() + 1
    random_state = np.random.RandomState(config['seed'])
    return n_items, random_state

def random_model_scoring(params, testset, testset_users):
    n_items, random_state = params
    scores = random_state.rand(testset_users, n_items)
    return scores

## Popularity-based

def build_popularity_model(config, trainset):
    item_popularity = trainset[trainset.anime_id.name].value_counts()
    return item_popularity

def popularity_model_scoring(params, testset, testset_users):
    item_popularity = params
    n_items = item_popularity.index.max() + 1
    n_users = testset_users
    
    # fill in popularity scores for each item with indices from 0 to n_items-1
    popularity_scores = np.zeros(n_items,)
    popularity_scores[item_popularity.index] = item_popularity.values
    
    # same scores for each test user
    scores = np.tile(popularity_scores, n_users).reshape(n_users, n_items)
    return scores

## Ploting

def plot_metric(metric_name, topn_grid, scores_list, holdout, anime):
    
    plt.figure(figsize=(10, 10))

    for model_id, (model_name, score) in enumerate(scores_list):

        plot_data = []
        
        for topn in topn_grid:
    
            recoms = simple_model_recom_func(score, topn)
            hr, mrr, cov = model_evaluate(recoms, holdout, anime, topn)

            if metric_name == 'mrr':
                val = mrr
            elif metric_name == 'hr':    
                val = hr
            elif metric_name == 'cov':
                val = cov
            else:
                raise ValueError
            
            plot_data.append(val)
            
        plt.plot(topn_grid, plot_data, '-o', label=model_name)
    
        
    plt.xlabel('topn')
    plt.title(metric_name.upper())
    
    plt.grid()
    plt.legend()
    plt.show()