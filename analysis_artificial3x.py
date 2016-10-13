from analysis_artificial import analysis_general


def do_analysis():
    N_features = [3,5,7,10]#,11,13,16]
    N_samples = 100
    analysis_general("artificial30",N_features, N_samples)
    analysis_general("artificial31",N_features, N_samples)
    analysis_general("artificial32",N_features, N_samples)
    analysis_general("artificial33",N_features, N_samples)
    analysis_general("artificial34",N_features, N_samples)
do_analysis()