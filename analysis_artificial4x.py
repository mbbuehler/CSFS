from analysis_artificial import analysis_general


def do_analysis():
    N_features = [3,5,7,10]#,11,13,16]
    N_samples = 100
    analysis_general("artificial40",N_features, N_samples)
    analysis_general("artificial41",N_features, N_samples)
    analysis_general("artificial42",N_features, N_samples)
    analysis_general("artificial43",N_features, N_samples)
    analysis_general("artificial44",N_features, N_samples)
do_analysis()