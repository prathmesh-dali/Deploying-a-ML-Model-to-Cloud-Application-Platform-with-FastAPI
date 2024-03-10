# Model Card

## Model Details

The model predivts if the persons salary is >50k or not based on the info. We trained the RandomForestClassifier with GridSearchCV to optimize the classifier hyperparameters from scikit learn 1.3.2. Optimal parameters used:
* n_estimators: 100
* max_depth: 10
* min_samples_split: 5

## Intended Use

This model can be used for predicting persons income range to provide some services based on the income range. 

## Training Data
The training data is available at [UCI Library](https://archive.ics.uci.edu/ml/datasets/census+income) as well the training data is provided in the git reposity [here](https://github.com/prathmesh-dali/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI/raw/master/data/cleaned_data.csv).

## Evaluation Data

The evaluation data was obtained by spliiting training data in 80:20 ratio.

## Metrics
Metrics used for eveluating model performance were:
* Precision: 0.766016713091922
* Recall: 0.5399214659685864
* Fbeta: 0.6333973128598848

## Ethical Considerations
The dataset should not be considered as a fair representation of the salary distribution and should not be used to assume salary level of certain population categories.

## Caveats and Recommendations
The dataset was collected in 1996 and is outdated. It is expected to be used for training ML classifiers or related problems. It does not represnt adequetly the recent salary distribution.