import numpy as np

# def cohen_d(x, y):
#     """
#     Calculates effect size (cohen's d)
#     source: http://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
#     :param x: list()
#     :param y: list()
#     :return: float
#     """
#     nx = len(x)
#     ny = len(y)
#     dof = nx + ny - 2
#     print(np.mean(x), np.mean(y), np.std(x), np.std(y), len(x), len(y))
#     return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def hedges_g(x, y):
    """
    Calculcates Hedge's g for lists / sets with unequal length
    source: https://en.wikipedia.org/wiki/Effect_size#Hedges.27_g
    :param x: list/set
    :param y: list/set
    :return: float
    """
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    std_x = np.std(x)
    std_y = np.std(y)
    n_x = len(x)
    n_y = len(y)
    pooled_s = np.sqrt(((n_x - 1) * std_x**2 + (n_y - 1) * std_y**2)/( n_x + n_y - 2))
    g = (mean_x - mean_y) / pooled_s
    return g

def test():
    x = [2,4,7,3,7,35,8,9]
    y = [i*2 for i in x]
    x.append(10)
    print(hedges_g(x, y))
    assert round(hedges_g(x,y),5) == round(-0.610541719138, 5)

if __name__ == '__main__':
    test()