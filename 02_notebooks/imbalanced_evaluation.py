import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import percentileofscore
warnings.filterwarnings("ignore")
from optbinning import scorecard
from sklearn.metrics import classification_report, confusion_matrix


def ks_thres(index: list, true: list, prob: list):
    pred = pd.DataFrame({'id': index, 'true':true, 'prob':prob})
    total_non_event = len(pred[pred['true']==0])
    total_event = len(pred[pred['true']==1])
    
    # Table
    threshold_prob = []
    non_event = []
    event = []

    thres_perc = [round(np.percentile(pred['prob'], p), 3) for p in list(range(0, 110, 10))]
    thres_perc = list(dict.fromkeys(thres_perc))
    for thres in thres_perc:
        threshold_prob.append(thres)
        non_event.append(len(pred[((pred['prob']<=thres) & (pred['true']==0))])/total_non_event)
        event.append(len(pred[((pred['prob']<thres) & (pred['true']==1))])/total_event)

    ks_df = pd.DataFrame({'threshold':threshold_prob, 'non_event':non_event, 'event':event})
    ks_df['KS'] = abs(ks_df['non_event'] - ks_df['event'])

    thres_base = list(ks_df[ks_df['KS']==ks_df['KS'].max()]['threshold'])[0]

    threshold_prob = []
    non_event = []
    event = []
    thres_perc2 = list(ks_df[ks_df['threshold']==thres_base].index)[0]*10
    thres_perc2 = list(range(thres_perc2-9, thres_perc2+10, 1))
    thres_perc2 = [round(np.percentile(pred['prob'], p), 3) for p in thres_perc2]
    thres_perc2 = list(dict.fromkeys(thres_perc2))
    for thres in thres_perc2:
        thres = thres
        threshold_prob.append(thres)
        non_event.append(len(pred[((pred['prob']<thres) & (pred['true']==0))])/total_non_event)
        event.append(len(pred[((pred['prob']<thres) & (pred['true']==1))])/total_event)

    ks_df2 = pd.DataFrame({'threshold':threshold_prob, 'non_event':non_event, 'event':event})
    ks_df2['KS'] = abs(ks_df2['non_event'] - ks_df2['event'])

    ks_df = pd.concat([ks_df, ks_df2], axis=0)
    ks_df = ks_df.sort_values('threshold')
    ks_df = ks_df.drop_duplicates(subset=['threshold']).reset_index(drop=True)

    ks_df['percentile'] = [round(percentileofscore(list(pred['prob']), x),4) for x in list(ks_df['threshold'])]
    ks_df['percentile'] = [round(percentileofscore(list(pred['prob']), x),4) for x in list(ks_df['threshold'])]
    ks_df = ks_df[['percentile', 'threshold', 'non_event', 'event', 'KS']]

    thres_x = list(ks_df[ks_df['KS']==ks_df['KS'].max()]['threshold'])[0]
    
    return thres_x, ks_df


def eval_report(index: list, true: list, prob: list, label:str, thres: float=None):
    print(label)
    pred = pd.DataFrame({'id': index, 'true':true, 'prob':prob})
    
    plt.figure(figsize=(16,10))
    plt.subplot(2, 3, 1)
    scorecard.plot_cap(pred['true'], pred['prob'])

    plt.subplot(2, 3, 2)
    scorecard.plot_ks(pred['true'], pred['prob'])

    plt.subplot(2, 3, 3)
    scorecard.plot_auc_roc(pred['true'], pred['prob'])

    # Distribution
    if thres is None:
        thres, ks_df = ks_thres(pred.index, pred['true'], pred['prob'])

    # Boxplot
    plt.subplot(2, 3, 4)
    sns.boxplot(data=pred, x='true', y='prob')
    plt.title('Boxplot')
    plt.xlabel('')
    plt.ylabel('Probabiltiy')
    plt.axhline(thres, color='tab:green', linestyle='--')

    # Count Plot
    plt.subplot(2, 3, 5)
    plt.hist(x=pred.loc[pred['true']==0, 'prob'], color='tab:blue', orientation='horizontal', alpha=0.3, label='good (0)')
    plt.hist(x=pred.loc[pred['true']==1, 'prob'], color='tab:orange', orientation='horizontal', alpha=0.3, label='bad (1)')
    plt.legend()
    plt.title('Count plot')
    plt.xlabel('Count')
    plt.axhline(thres, color='tab:green', linestyle='--')

    # Density Plot
    plt.subplot(2, 3, 6)
    plt.hist(x=pred.loc[pred['true']==0, 'prob'], density=True, color='tab:blue', orientation='horizontal', alpha=0.3, label='good (0)')
    plt.hist(x=pred.loc[pred['true']==1, 'prob'], density=True, color='tab:orange', orientation='horizontal', alpha=0.3, label='bad (1)')
    plt.legend()
    plt.title('Density plot')
    plt.xlabel('Density of each class (%)')
    plt.axhline(thres, color='tab:green', linestyle='--')
    plt.show()

    # Prediction Result
    print(f'Probability threshold: {thres}')
    print('confusion_matrix')
    print(pd.DataFrame(confusion_matrix(pred['true'], pred['prob']>thres)))
    print('')
    print('Classification Report')
    print(classification_report(pred['true'], pred['prob']>thres))