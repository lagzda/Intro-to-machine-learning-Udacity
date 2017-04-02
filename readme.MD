# Intro to Machine learning Final Project

### Identify fraud from Enron Email

1. **Understanding the dataset and question**

In 2002 one of the biggest frauds in American history was discovered. Enron, a multi billion dollar energy trading company went bankrupt in less than 2 years. This was due to some corrupted directors and employees who made it seem that the company was running perfectly when in reality it was crashing. The aim of the project is to build a classifier that will be able to identify wether a person is a POI (Person of Interest related to the fraud) and who was not.

The dataset I will be analysing consists of 146 entries of Enron employees with included financial information and email information. In addition to that they have been labeled as a POI or not.

The exploration step will greatly help me understand the data, gain some intuition about trends, outliers and I will be able to make initial guesses for feature selection, combination and algorithm selection.

_Total entries_: 146

_Number of POIs_: 18 (~12%)

_Initial number of features_: 21

_Features with most NaN values_:

- Loan advances - 142
- Director fees - 129
- Restricted stock deferred - 128
- Deferral payments - 107

_POI features with most NaN values_:

- Loan advances - 18
- Director fees - 18
- Restricted stock deferred - 17
- Deferral payments - 13

_Initial score of Naive Bayes classifier on training data_: .151724137931

_Initial score of Naive Bayes classifier on testing data_: .136363636364

It is clearly visible that some of the features have a very big number of NaN values which just means that the values are missing. This mostly implies that theses features may not be useful to include when training the classifier, especially when some features don&#39;t describe POIs at all. Also the fact that this dataset does not contain a lot of POIs (only 12%) means I will have to pay more attention when evaluating the performance of the algorithm. As a start this step greatly helped to identify the first steps of feature selection. For benchmarking purposes I include the initial scores of the classifier and will keep documenting this after each important step.

2. **Feature selection / engineering**

The first thing I wanted to do is visualise the data so I could more intuitively grasp how the features are correlated and to see if there aren&#39;t any outliers. For that I need my initial feature list but since I already identified that some features have too many NaN values (at least more than half) they would automatically be taken out. Also by intuition the emails column was taken out as there were couple of missing values and just logically thinking I could not come up with a way it would influence the decision process. I created a 2x4 plotter which plots a specified feature on the x axis and all the remaining features in each plot&#39;s y axis.

With salary put on x axis several plots produced a big outlier. It was easily found that it was the total of a particular feature and was not necessary in fact harmful for the analysis so I removed it.

After the first outlier removed I put the remaining features on the x axis and checked for outliers.

While looking at the data a discovery was made that John Lavorato has received the biggest bonus and has exchanged the most emails with POIs but research on the internet did not provide sufficient evidence that this person was a POI. Other than that nothing else was discovered.

By removing the single outlier the score of the classifier increased dramatically.

_Naive Bayes classifier on training data after removing outlier_: .847222222222

_Naive Bayes classifier on testing data after removing outlier_: .863636363636

By looking at the emails it seemed reasonable to create two new features that would represent the ratio of how much person was in contact with a POI relative to other people. The reasoning behind this choice was that it could more correctly represent which people were in close contact with POIs. When implemented the two new features did not affect the score at all but they will be also tested in the final classifier to determine whether to use them or not.

3 feature selection algorithms were run initially to see which features the algorithms think are most important. This will help me eliminate redundant features, decrease the complexity of the algorithm, and finally utilise only the features that explain most of the variance. The algorithms were:

- SelectKBest
- Decision Tree feature importances
- Randomized linear regression

Most consistent and strongest features were - bonus, expenses, from messages, long term incentive and to messages. To see how the feature selection affects the classifier I put SelectKBest selector in my classifier pipeline to only select the best 5 features. Both training and testing scores improved.

_Naive Bayes classifier on training data after feature selection_: .875

_Naive Bayes classifier on testing data after feature selection_: .886363636364

Principal Component Analysis was introduced in the classifier pipeline as well. It was tested alone and in combination with feature selection with reduction to 3, 5 and 10 dimensions out of total of 15. The scores matched the previous score at best so I chose to leave PCA out for now to keep the data understandable.

3. **Picking and tuning of a classifier**

I have chosen to test three classifiers:

- Gaussian Naive Bayes classifier
- Random forest classifier
- AdaBoost

All of the classifiers will be tested with the transformation of the features with the SelectKBest method of 5 best features.

Firstly the results with default parameters for the classifiers.

_Gaussian Naive Bayes:_

- _Score with train and then test samples -_ .84, .909090909091
- _Precision, Recall, F1 -_ .6, .6, .6

_Random Forest:_

- _Score with train and then test samples -_ .97, .886363636364
- _Precision, Recall, F1 -_ .5, .2, .285714285714

_Adaboost:_

- _Score with train and then test samples -_ 0, 0.818181818182
- _Precision, Recall, F1 -_ .2, .2, .2

It seems that out of the box performance is best for the Naive Bayes classifier although I have to consider that since the dataset contains a very small sample of POIs the predictions could just be lucky guesses. That could also be reflected in the relatively low training score. The evaluation method also is a bit concerning at the moment because it just performs one train / test split and while it has at least 6 POIs in the test split, these could be the biggest outliers that define what a POI is while the algorithm is trained on subtle POIs. This concern initially will be addressed in the algorithm tuning step. Grid Cross Validation will be used to find the best performing algorithm and it&#39;s parameters. Grid search is important to optimise the performance of the algorithm and avoid overfitting. A very good thing about the GridCV is that it evaluates the performance using cross validation. That means the algorithm will use each partitioned (the number is determined by specifying how many &quot;folds&quot; are necessary) data chunk both as a training and testing set.

I will use the default CV value of 3 to evaluate the performance.

The parameters used for each classifier are as follows:

_Gaussian Naive Bayes:_

- _Priors:_ [.875, .125], [.9, .1], [.95, .05]

_Random Forest:_

- _No of_ _estimators__:_ [10, 50, 100, 200]
- _Criterion :_ (gini, entropy)
- _Max_ _depth:_ [None, 4, 10, 50, 100]
- _Min samples split:_ [2, 50, 100]

_Adaboost:_

- _No of estimators_: [50, 100, 200]
- _Algorithm_: (SAMME, SAMME.R)
- _Learning Rate_: [.1, 1,10,100]

The choice of tuning the priors for the NB classifier arose when thinking about the dataset and the final outcome. Ultimately I found that it would be more disastrous if innocent people would be convicted rather than missing a few fraudsters. With Random Forest and Adaboost parameters I mostly was concerned with just testing out the parameters that I thought could make the biggest difference.

 The best parameters and scores for each classifier are as follows:

_Gaussian Naive Bayes_:

- _Priors_: [.9, .1]
- _Score with train and then test samples_ - .86, .909090909091
- _Precision, Recall, F1_ - .6, .6, .6

_Random Forest:_

- _Default parameters_
- _Score with train and then test samples_ - .97, .886363636364
- _Precision, Recall, F1_ - .5, .2, .285714285714

_Adaboost:_

- _No of estimators_: 50, _Learning rate_: 1, _Algorithm_: SAMME
- _Score with train and then test samples_ - .87, .886363636364
- _Precision, Recall, F1_ - 0, 0, 0

Many of the parameter change effects I could not explain. Especially for the Adaboost where the precision and recall scores dropped to 0. I started to suspect that it might have something to do with an incompetent testing method which will be looked at the later stage. The Naive Bayes worked perfectly as I suspected especially with the adjusted priors to lean towards less false positives. Another very interesting discovery made by doing research was to build a pipeline for the Naive Bayes algorithm. I decided to implement Standard Scaler which distributes z-scores because the ranges for salary and email counts for example are drastically different. Then I could bring back the principal component analysis to reduce the dimensionality. Finally I select the 5 features which explain the most variance and adjust the priors to an extreme of [0.99, 0.01] to truly try to hit the right POIs. And the results were surprisingly good:

Gaussian Naive Bayes with feature scaling and PCA:

- _Score with train and then test samples_ - .88, .909090909091
- _Precision, Recall, F1_ - .6666, .4, .5
- _True negatives, False negatives, True positives, False positives_ - 38, 3, 2, 1

Although the recall and F1 scores dropped I felt that this was a big improvement, because as I mentioned before to me it&#39;s more important not to accuse innocent people. And actually when testing on the official tester in the &quot;tester.py&quot; file it showed me the best performance of all even for recall. This confirmed my doubts about the current testing method I am using at the moment. Also for the first time I printed the confusion matrix and noticed that the results can be easily skewed with such a minimal number of POIs.

4. **Evaluation**

The current method of evaluating the algorithms is a train test split on the data. The test sample size is 0.3 of the total which is about 44 entries. Firstly since there is not a lot of data this will leave out a lot of data from training, and more importantly almost half of the POIs will be in the testing set so the algorithm could have a huge bias error. Evaluating is important for many things. Firstly to see how the algorithm is performing in generally (accuracy), but many things can be measured such as efficiency (time), overfitting and / or underfitting, the algorithm could be even skewed to perform more in a certain way as is my case where I try to eliminate false positives. Choosing the right evaluation method is very important and mostly depends on the data. Because I have very a limited dataset my initial guess is that a k-fold cross validation with shuffle would be the best choice so that the algorithm gets trained on all partitions of the data and all partitions get tested as well.

When I finally tested the classifier with k-fold cross validation I immediately realised that this method will not work because the dataset is so small the results vary a lot and are not really representative of what the actual results really mean.

I only came up with two solutions that were possible to increase the relevance of the evaluation. One is to carefully select the splits and really pay attention to what is measured in the specific split. The other is doing the stratified k fold. Just from researching the method it seems that it would be best for such a dataset where instances of one class are a lot more than the other. Because stratified k fold is already implemented in the tester code I will just use that to evaluate my metrics. Also the created new features mentioned before did not affect the score at all so they will not be included in the training.

Final classifier pipeline and it&#39;s results using the provided tester code

1. Standard scaler
2. PCA (10 dimensions)
3. Select K Best (5)
4. Naive Bayes with priors = [.99, .01]

- _Score_ - .86513
- _Precision_ - .49178
- _Recall_ - .34400
- _F1_ - .40482

In conclusion this classifier would never work in real life because it still identifies more innocent people as guilty. I would argue that the problem is within the data and how I have chosen to preprocess it. Obviously for more accurate results more POIs would have to be introduced and possibly even more features that would discriminate between the guilty and innocent class. I should also do additional research on how to look at data and how to properly process particular pieces of information. Still to end on a good note the required threshold for completing the project was met and it was a really good introductory course on machine learning.