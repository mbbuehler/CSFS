
Expert estimates for feature relevance are imperfect
===================
A crowdsourcing approach to estimate feature relevance before obtaining data.

Keywords
-------------
Feature Ranking, Feature Selection, Crowdsourcing, Wisdom of Crowds, Machine Learning.

Functional Prototype
-------------
A working prototype is available [here](http://mbuehler.ch/krowdd). For more details, see Section "Web Application" bellow or download the paper on the [UZH website](https://www.merlin.uzh.ch/publication/show/15636).

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

Web Application
-------------
Try KrowDD online! 
On http://mbuehler.ch/krowdd you can upload your own data and obtain a relevance estimation for each feature. 
You only need the following:
- Job title: the title of the job
- E-mail: your email
- AMT access key ID and AMT secret access key: your Amazon Mechanical
Turk access key ID and the corresponding secret access key. These credentials are
required to collect the crowd estimates on Amazon Mechanical Turk.
- CSV file: a CSV file with seven columns: Feature, Question P (X|Y = 0), Question
P (X|Y = 1), Question P (X), P (X|Y = 0), P (X|Y = 1), P (X). Users need to fill
at least one field for each sibling (e.g. Question P (X|Y = 1) and P (X|Y = 1)). For
each feature, the user can either provide descriptions for the (conditional) means
of this feature or directly enter a value. KrowDD only queries feature means for
which no value has been provided.
- Target mean (optional): you can decide between defining a target mean (e.g. if it is
already known that the target variable is balanced, the user might use a target
mean of 0.5) or querying the target mean from the crowd.

The estimates acquired from the crowd and the feature data (Information Gain and probability
estimates) can be downloaded as CSV files.

Questions / Feedback
-------------
For questions / feedback, don't hesitate and contact me.

Screenshots
-------------
![New Job View](/app_fred/screenshots/new_job.png "New Job View")
![Job Status View](/app_fred/screenshots/job_status.png "Job Status View")
![Job Result View](/app_fred/screenshots/job_result.png "Job Result View")



