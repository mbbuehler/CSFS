KrowDD: Estimating the Usefulness of a Feature before Obtaining Data for It
===================
A crowdsourcing approach to estimate feature relevance before obtaining data.
Keywords
-------------
Feature Ranking, Feature Selection, Crowdsourcing, Wisdom of Crowds, Machine Learning.

Tool
-------------
The tool will soon be available online for anyone to use. Hang on some longer!
Datasets (Preprocessed)
-------------
The preprocessed datasets are stored as .csv here: [paper_plots-and-data/datasets](paper_plots-and-data/datasets)

AUC for Each Condition
-------------
This data was used to create Tables 2, 3 and 4, as well as Figures 2, 3 and 4. The calculated AUC scores (Naive Bayes) for each conditions are stored as .json here: [paper_plots-and-data/datasets](paper_plots-and-data/datasets)
The files are structured as follows:
{CONDITION: {NUMBER_OF_FEATURES: [AUC1, AUC2, AUC3,...],...},...}
Where Condition is one of the following:
- 'Data Scientists': Data Scientists from [Upwork](https://www.upwork.com/).  There is one AUC score per number of features for each expert.
- 'Domain Experts': Manually recruited Domain Experts. There is one AUC score per number of features for each expert.
- 'KrowDD': AUC scores for the approach proposed here.
- 'Laypeople': AUC scores for rankings from [Amazon Mechanical Turk](https://www.mturk.com/mturk/welcome)
- 'Random': AUC scores for a random selection of features.
- 'Actual': AUC scores when using the actual Information Game Ranking from the dataset.
- 'Best': Best possible* AUC score for chosen Classifier (Naive Bayes)
- 'Worst': Worst possible* AUC score for chosen Classifier (Naive Bayes)

> *Please note that the AUC score has been calculated as the mean of a 10-fold cross validation. Therefore, it is possible that this score is slightly higher/lower than the true best score.

NUMBER_OF_FEATURES denotes the number of features used to train the classifier. For example, the AUC scores for NUMBER_OF_FEATURES=3 have been calculated using the best three features according to the given condition.


Figure 5: Crowd Estimate Errors
-------------
Data for creating [Figure 5](paper_plots-and-data/fig5_no_answers_vs_delta.png) can be found as .json [here](paper_plots-and-data/fig5_no_answers_vs_delta.json).
The File is structured as follow:
{NUMBER_OF_ANSWERS:{INDEX_0: Delta_0, INDEX_1: Delta_1,...},...}
NUMBER_OF_ANSWERS denotes the numbers of estimates sampled from all acquired crowd estimates (without replacement). DELTA_X denotes the average of the absolute difference between the means calculated from the actual dataset and the aggregated crowd estimates. INDEX_X is an internal index.





