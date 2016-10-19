import os
import pickle
import re
import sys
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from CSFSEvaluator import CSFSEvaluator
from CSFSLoader import CSFSLoader
from CSFSSelector import CSFSBestUncertainSelector

def _conduct_analysis(df, target, std, N_features, N_samples, dataset_name):
    sys.stdout.write('std:{}{}'.format(std,'\n'))
    evaluator = CSFSEvaluator(df, target, fix_std=std)
    best_noisy_selector = CSFSBestUncertainSelector(df, target, fix_std=std)
    for n in N_features:
        aucs = evaluator.evaluate_noisy(n, N_samples, best_noisy_selector)

        filepath = '{}/{}features_{}samples_{:.9f}std'.format(dataset_name, n, len(aucs['best_noisy']), std)
        if not os.path.isdir('pickle-dumps/'+dataset_name):
            os.mkdir('pickle-dumps/'+dataset_name)

        pickle.dump(aucs, open("pickle-dumps/{}.pickle".format(filepath), 'wb'))

def analysis_general(dataset_name, N_features, N_samples):
    path = "datasets/artificial/{}.csv".format(dataset_name)
    df = CSFSLoader().load_dataset(path)
    target = "T"

    Parallel(n_jobs=8)(delayed(_conduct_analysis)(df, target, std, N_features, N_samples, dataset_name) for std in np.linspace(0.00001, 0.3, 500))

def get_result_data(n_features, dataset_name, N_samples=100):
    """

    :return: {no_features: {std: auc},...} e.g. {16: {0.200036667: 0.53119531952662713, 0.105176567: 0.57273262130177505
    """
    path = 'pickle-dumps/{}/'.format(dataset_name)
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]

    results = dict()
    pattern = r'(\d+)features_{}samples_(.*?)std'.format(N_samples)
    for f in files:
        match = re.match(pattern, f)
        # print(f)
        no_features = int(match.group(1))
        std = float(match.group(2))

        if no_features not in results.keys():
            results[no_features] = dict()

        results[no_features][std] = (np.mean(pickle.load(open(os.path.join(path,f), 'rb'))['best_noisy']))
    if n_features:
        results = {r:results[r] for r in n_features}
    return results

def extract_x_y(result, n_features, start_lim=0):
    """
    extracts x and y from results for a certain n_features
    :param result:
    :param n_features:
    :return: x,y where x: std and y:auc
    """
    if n_features not in result.keys():
        print('{} not found'.format(n_features))
        return None
    x = sorted([std for std in result[n_features].keys() if std > start_lim])
    y = [result[n_features][std] for std in x]

    return np.array(x, dtype=float), np.array(y, dtype=float)

def visualise_results(dataset_name, N_features, fit_curve=False, start_lim=0, show_plot=False, N_samples=100):
    results = get_result_data(N_features, dataset_name, N_samples)
    plt.hold(True)
    params = dict()

    def func(x, w1, p1, w2, p2):
        return w1 * pow(x, p1) + w2 * pow(x, p2)

    # def func(x, w1, p1, w2, p2, w3):
    #     return w1 * pow(x, p1) + w2 * pow(x, p2) + w3 * np.log10(x)

    for n_f in N_features:
        print('== no of features: {}'.format(n_f))
        x,y = extract_x_y(results, n_f, start_lim=0)
        std = np.std(y)
        plt.plot(x, y, alpha=0.5, label='data {}'.format(n_f))
        if fit_curve:
            x,y = extract_x_y(results, n_f, start_lim=start_lim)
            try:
                popt, pcov = curve_fit(func, x, y)
                params[n_f] = popt
                perr = np.sqrt(np.diag(pcov))
                avg_err = np.mean(perr)
                print('params: {} '.format(popt))
                print('errors: {}'.format(perr))
                print('avg error: {}'.format(avg_err))

                plt.plot(x, func(x, *popt), '-k', linewidth=1, label="Fitted {} (avg err: {:.3f})".format(n_f, avg_err))
            except:
                print('no matching curve found')

    plt.legend(loc=3)
    plt.title('{}: AUC for noisy IG'.format(dataset_name))
    plt.xlim([-.01, .3])
    plt.ylim([0.5, 1.05])
    plt.xlabel('std')
    plt.ylabel('auc')
    fig1 = plt.gcf()
    if show_plot:
        plt.show()


    dataset_class = dataset_name
    if not os.path.isdir('plots/{}/'.format(dataset_class)):
            os.mkdir('plots/{}/'.format(dataset_class))
    fig1.savefig('plots/{}/{}.png'.format(dataset_class, dataset_name), dpi=100)
    plt.hold(False)
    plt.clf()

def evaluate():
    #todo<
    N_features = [3,5,7,10]
    dataset_names = ['artificial20','artificial21','artificial22','artificial23','artificial24','artificial25','artificial26','artificial27']
    Parallel(n_jobs=3)(delayed(visualise_results)(dn, N_features, False) for dn in dataset_names)

if __name__ == "__main__":

    # do_analysis()
    evaluate()