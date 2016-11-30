import os
from abc import abstractmethod, ABCMeta

import pandas as pd
import numpy as np
import sys


class KnapsackSolver():
    """
    http://codereview.stackexchange.com/questions/20569/dynamic-programming-solution-to-knapsack-problem (accessed 30.11.16)
    """
    @staticmethod
    def isinteger(x):
        """
        :param x: single value or list
        :return: boolean True if all values are int, else False
        """
        return np.equal(np.mod(x, 1), 0).all()

    @staticmethod
    def knapsack(items, maxweight):
        weights = [items[i][1] for i in range(len(items))]
        if not KnapsackSolver.isinteger(weights):
            print('Weights have to be integers. Aborting knapsack.', file=sys.stderr)
            return None

        # Create an (N+1) by (W+1) 2-d list to contain the running values
        # which are to be filled by the dynamic programming routine.
        #
        # There are N+1 rows because we need to account for the possibility
        # of choosing from 0 up to and including N possible items.
        # There are W+1 columns because we need to account for possible
        # "running capacities" from 0 up to and including the maximum weight W.
        bestvalues = [[0] * (maxweight + 1)
                      for i in range(len(items) + 1)]

        # Enumerate through the items and fill in the best-value table
        for i, (value, weight) in enumerate(items):
            # Increment i, because the first row (0) is the case where no items
            # are chosen, and is already initialized as 0, so we're skipping it
            i += 1
            for capacity in range(maxweight + 1):
                # Handle the case where the weight of the current item is greater
                # than the "running capacity" - we can't add it to the knapsack
                if weight > capacity:
                    bestvalues[i][capacity] = bestvalues[i - 1][capacity]
                else:
                    # Otherwise, we must choose between two possible candidate values:
                    # 1) the value of "running capacity" as it stands with the last item
                    #    that was computed; if this is larger, then we skip the current item
                    # 2) the value of the current item plus the value of a previously computed
                    #    set of items, constrained by the amount of capacity that would be left
                    #    in the knapsack (running capacity - item's weight)
                    candidate1 = bestvalues[i - 1][capacity]
                    candidate2 = bestvalues[i - 1][capacity - weight] + value

                    # Just take the maximum of the two candidates; by doing this, we are
                    # in effect "setting in stone" the best value so far for a particular
                    # prefix of the items, and for a particular "prefix" of knapsack capacities
                    bestvalues[i][capacity] = max(candidate1, candidate2)

        # Reconstruction
        # Iterate through the values table, and check
        # to see which of the two candidates were chosen. We can do this by simply
        # checking if the value is the same as the value of the previous row. If so, then
        # we say that the item was not included in the knapsack (this is how we arbitrarily
        # break ties) and simply move the pointer to the previous row. Otherwise, we add
        # the item to the reconstruction list and subtract the item's weight from the
        # remaining capacity of the knapsack. Once we reach row 0, we're done
        reconstruction = []
        i = len(items)
        j = maxweight
        while i > 0:
            if bestvalues[i][j] != bestvalues[i - 1][j]:
                reconstruction.append(items[i - 1])
                j -= items[i - 1][1]
            i -= 1

        # Reverse the reconstruction list, so that it is presented
        # in the order that it was given
        reconstruction.reverse()

        # Return the best value, and the reconstruction list
        return bestvalues[len(items)][maxweight], reconstruction


class Recommender:
    """
    Nachdem für jedes Feature der IG berechnet wurde, kann Hans ein Budget angeben und berechnen, welche Feature-Auswahl
     für dieses Budget wahrscheinlich optimal ist. Hans kann sehr einfach verschiedene Budget’s prüfen.
    """

    def __init__(self):
        pass

    def _get_recommended_features(self, df_cost_ig, reconstruction):
        features_recommended = list()
        for selected in reconstruction:
            ig = selected[0]
            cost = selected[1]
            features_recommended.append(df_cost_ig[(df_cost_ig['IG'] == ig) & (df_cost_ig['cost']==cost)]['feature'].values[0])
        return features_recommended

    def doit(self, df_cost_ig, budget):
        costs = df_cost_ig['cost'].values
        values = df_cost_ig['IG'].values

        items = [[values[i], costs[i]] for i in range(len(costs))]
        bestvalue, reconstruction = KnapsackSolver.knapsack(items, budget)

        features = self._get_recommended_features(df_cost_ig, reconstruction)

        return bestvalue, features


def test():
    data = {
            'feature': ['F1', 'F2', 'F3', 'F4'],
            'cost': [5, 4, 6, 3],
            'IG': [10, 40, 30, 50]
            }
    df_cost_ig = pd.DataFrame(data)
    bestvalue, features_recommended = Recommender().doit(df_cost_ig, 10)
    assert bestvalue == 90
    assert features_recommended == ['F2', 'F4']


if __name__ == '__main__':
    test()