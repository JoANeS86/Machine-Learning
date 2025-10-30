## The Nuts and Bolts of Machine Learning

#### <ins>The different types of Machine Learning</ins>

  - Supervised Machine Learning: Uses labeled datasets to train algorithms to classify or predict outcomes.

<p align="center">
  <img src="https://github.com/user-attachments/assets/16d38638-df43-42e7-94c7-e3157741294d" />
</p>

To summarize, supervised machine learning algorithms use data with answers already in it, and use it to make more answers either by categorizing or by
estimating future data.

  - Unsupervised Machine Learning: Uses algorithms to analyze and cluster unlabeled datasets.

<p align="center">
  <img src="https://github.com/user-attachments/assets/887baef6-1fbf-4507-b275-5414bd84db9f" />
</p>

There are a couple of other types of machine learning besides supervised and unsupervised, like **Reinforcement Learning** or **Deep Learning**.

ML and AI refer to the same principle; training a computer to detect patterns in data without being explicitly
programmed to do so.

Supervised learning models that make predictions that are on a continuum are called **regression algorithms**.

    Continuous variables: Variables that can take on an infinite and uncountable set of values.
    
    Categorical variables: Variables that contain a finite number of groups or categories.
    
    Discrete features: Features with a countable number of values between any two values.

**Discrete variables are able to be counted, and categorical variables are able to be grouped**.

Knowing what type of features you have in a dataset, and what outcomes you're looking for, will help you to determine the most applicable ML model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/dfd94014-2746-4de8-81f3-ff3e9fddf62d" />
</p>

  - Recommendation Systems: Unsupervised learning techniques that use unlabeled data to offer relevant suggestions to users. The main goal of a
Recommendation System is to quantify how similar one thing is to another.

One approach used by Recommendation Systems is Content-based filtering, where comparisons are made based on attributes of content.

Another one is, Collaborative filtering, where comparisons are made based on who else liked the content (the Recommendation System doesn't need to 
know anything about the content itself).

Natural Language Processing, or NLP, could be used too (document-term matrix: A mathematical matrix that describes the frequency of terms that occur 
in each document in a collection.)

Build ethical models

<p align="center">
  <img src="https://github.com/user-attachments/assets/e3b3fe71-1ec5-4270-9207-10da2a798be5" />
</p>

  - Integrated Development Environment (IDE): A piece of software that has an interface to write, run, and test a piece of code.

The two main types of Python files are known as Python scripts and Python notebooks.

    Python scripts are better for production-grade code, and are easier to debug and manage.

    Python notebooks are better for exploratory analyses, presentations, or anything that needs to be human-facing. You are
    able to insert images, text, and links directly into the code.

More about Python packages

<p align="center">
  <img src="https://github.com/user-attachments/assets/3d70a59b-de09-4f30-8129-a8d8a71a63e0" />
</p>

Generally, there are 3 types of Python packages:

  - Operational packages: They load, structure and prepare the dataset for further analysis (E.g., Pandas, Numpy, Scipy).
  - Data Visualization packages: They allow you to create plots and graphics of data (E.g., Matplotlib, Seaborn, Plotly).
  - Machine Learning packages: They give many functions to help build models from a dataset, along with functionality to examine
    the model once it has been built (E.g., Sklearn).

Apart from these, Python has thousands and thousands of packages publicly available.

#### <ins>Workflow for building complex models</ins>

  - Plan

In this section, the course is focused on ML, so it'll continue with an example related to it. But putting that apart, you need to think about whether you need a model in the first place! Many analytical tasks do not
require the creation of a model, and you could spend time creating something that is not necessary to what you’re trying to achieve.

Let's continue with ML and knowing what you need for a problem: The first thing to do when forming your plan is to consider the end goal, and something that can be determined immediately is what type of machine learning
model you’ll need.

    Supervised models are used to make predictions about unseen events. These types of models use labeled data, and the model
    will use these labels and the predictor variables present to learn from the dataset. And when given new data points,
    they’re able to make a prediction of the label.

    Unsupervised models, on the other hand, don’t really make predictions. They are used to discover the natural structure
    of the data, finding relationships within unlabeled data.

  - Analyze

The main focus of the analyze stage is to develop a deeper understanding of the data, while keeping in mind what the model needs to eventually predict.

**Feature engineering** is the process of using practical, statistical, and data science knowledge, to <ins>select, transform, or extract</ins> characteristics, properties, and attributes, from raw data.

    Feature selection: The goal of this type of feature engineering is to select the features in the data that contribute the
    most to predicting your response variable. In other words, you drop features that do not help in making a prediction.
    
    Feature transformation: In feature transformation, data professionals take the raw data in the data set and create features
    that are suitable for modeling. This process is done by modifying the existing features in a way that improves accuracy when
    training the model (Transformation = Change the existing features).

    Feature extraction: Taking multiple features to create a new one that would improve the accuracy of the algorithm
    (Extraction = Create new features from the data).

Datasets used in the workplace can sometimes require multiple rounds of EDA and feature engineering to get everything in a suitable format to train a model.

Class imbalance: When a dataset has a predictor variable that contains more instances of one outcome that another.

There are two general strategies to balance a dataset, and the method that is better to use generally is decided by how much data you have in the first place: Downsampling and Upsampling.

    Downsampling: Downsampling is the process of making the minority class represent a larger share of the whole dataset simply
    by removing observations from the majority class. It is mostly used with datasets that are large. But how large is large
    enough to consider downsampling? Tens of thousands is a good rule of thumb, but ultimately this needs to be validated by
    checking that model performance doesn’t deteriorate as you train with less data. 

    Upsampling: Upsampling is basically the opposite of downsampling, and is done when the dataset doesn’t have a very large
    number of observations in the first place. Instead of removing observations from the majority class, you increase the number
    of observations in the minority class. 

In both cases, upsampling and downsampling, it is important to leave a partition of test data that is unaltered by the sampling adjustment. You do this because you need to understand how well your
model predicts on the actual class distribution observed in the world that your data represents. In the case of the spam detector example, it’s great if your model can score well on resampled data
that is 80% not spam and 20% spam, but you need to know how it will work when deployed in the real world, where spam emails are much less frequent. This is why the test holdout data is not rebalanced.

Class rebalancing should be reserved for situations where other alternatives have been exhausted and you still are not achieving satisfactory model results (class imbalance isn't always
a problem).

  - Construct

Here, we're bringing the model to life: In this case, we're building a model called Naive Bayes.

    Naive Bayes: A supervised classification technique that is based on Bayes' Theorem with an assumption
    of independence among predictors.

**Naive Bayes** is a <ins>supervised classification</ins> technique based on Bayes' theorem with an assumption of independence among predictors. The effect of the value of a predictor variable on a given class is not affected by the values of other predictors. Let's break it down. Bayes' theorem gives us a method of calculating the **<ins>posterior probability</ins>**, which is the likelihood of an event occurring after taking into consideration new information.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c04c7bc7-2768-4881-93ad-2a8a90f60d22" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/26db510d-4b51-46a4-94c8-24416f762a03" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/17965dc2-0e32-4461-8856-958d6fcf033d" />
</p>

The process of finding the posterior probability needs to be done for every possible class that is potentially being predicted. In this case, there are only two outcomes; play or don't play.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a50ee723-7c90-48c0-92c4-68d50e51ba39" />
</p>

There are several implementations of Naive Bayes in scikit-learn, all of which are found in the **sklearn.naive_bayes** module.

When using Python to construct and test a model, we'll use the term "**fitting a model**", which means learning the relationship between inputs and outputs using training data 

    .fit(X, y)   Train the model using the input (X) and target (y) data

    .predict(X)  Use the trained model to make predictions on new input data (X)


# Add here the summary of "More about evaluation metrics for classification models".

**ROC** curves make it easy to identify the best threshold for making a decision, and the **AUC** can help you decide which categorization method is better.

#### <ins>Unsupervised learning techniques</ins>

K-means is an unsupervised learning algorithm that partitions a dataset into K distinct clusters by iteratively assigning data points to the nearest cluster centroid and updating the centroids to the mean of their assigned points. The goal is to minimize the variance within each cluster, grouping similar data points together. The process requires specifying the number of clusters (K) beforehand and is sensitive to initial centroid placement and outliers.

    Centroid: The center of a cluster determined by the mathematical mean of all the points in that cluster.

Something to be mindful of is that it's important to run the model with different starting positions for the centroids. This helps avoid poor clustering caused by <ins>local minima</ins>. In other words, not having an appropriate distance between clusters (Many ML implementations have removed this requirement, for example, in scikit-learn, **K-means++** helps to ensure that centroids aren’t initially placed very close together, which is when convergence in local minima is most likely to occur. 

K-means works by minimizing intercluster variance. In other words, it aims to minimize the distance between points and their centroids. This means that K-means works best when the clusters are round. If you aren’t satisfied with the way K-means is clustering your data, don’t worry, there are many other clustering methods available to choose from.

    DBSCAN

    Agglomerative Clustering

    Other clustering altorithms

Key metrics for representing K-means clustering

    Inertia: Sum of the squared distances between each observation and its nearest centroid.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b2de4c1a-ebae-4080-b2de-ca33409c0053" />
</p>

To evaluate Inertia, we use the elbow method.

Remember, you want inertia to be low, but if you add more and more clusters with only minimal improvement to inertia, you’re only adding complexity without capturing real structure in the data.

    Silhouette score: The mean of the silhouette coefficients of all the observations in the model.

<p align="center">
  <img src="https://github.com/user-attachments/assets/76825b12-7d88-4f4d-8e44-35247e214be3" />
</p>

A silhouette coefficient can range between -1 and +1. A value closer to +1 means that a point is close to other points in its own cluster and well separated from points in other clusters.

Note that, unlike inertia, silhouette coefficients contain information about both intracluster distance (captured by the variable a) and intercluster distance (captured by the variable b).

Use these metrics together to help inform your decision on which model to select.

#### <ins>Tree-based modeling</ins>

<ins>Tree-based modeling</ins>: Flow-chart-like supervised classification model and a representation of various solutions that are available to solve a given problem based on the possible outcomes of related choices. 

  - Explore decision trees

<p align="center">
  <img src="https://github.com/user-attachments/assets/d0e9f180-2cdc-45d1-add3-d6d63a4f409e" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/8f074185-26b3-4ff8-a753-6a29c122c23c" />
</p>

To choose a split, we use Gini impurity, and calculate the weighted average of Gini impurities.

Hyperparameter tuning

    - Hyperparameters: Parameters that can be set before the model is trained, like Max depth or Min samples leaf.

    - GridSearch: A tool to confirm that a model achieves its intended purpose by systematically checking every
    combination of hyperparameters to identify which set produces the best results based on the selected metric.

Verify performance using validation

    - Model validation is the whole process of evaluating different models, selecting one, and then continuing
    to analyze the performance of the selected model to better understand its strengths and limitations. (Note
    that each unique combination of hyperparameters is a different model).

    - Validation can be performed using a separate partition of the data, or it can be accomplished with cross-validation
    of the training data, or both.

    - Cross-validation splits the training data into k number of folds, trains a model on k – 1 folds, and uses the fold
    that was held out to get a validation score. This process repeats k times, each time using a different fold as the
    validation set. 

    - Cross-validation is more rigorous, and makes more efficient use of the data. It’s particularly useful for
    smaller datasets.

    - Validation with a separate dataset is less computationally expensive, and works best with very large datasets.

    - For a truly objective assessment of model performance on future data, the test data should not be used to select
    a final model.

Bootstrap aggregation

    - Ensemble learning (or "ensembling"): Involves building multiple models and then aggregating their outputs to make
    a prediction.

    - Base learner: Each individual model that comprises an ensemble.

When Bagging is used with Decision Trees, we get a **Random Forest**: Ensemble of decision trees trained on bootstrapped data with randomly selected features.

Random Forest models have the same hyperparameters as Decision Trees (since they're ensembles of them), but random forests also have some other hyperparameters which control the ensemble itself, like <ins>max_features</ins> or <ins>n_estimators</ins>.

Introduction to boosting

    - Boosting: Supervised learning technique that builds an ensemble of weak learners sequentially, with each
    consecutive learner trying to correct the errors of the one that preceded it.

    - Adaptive Boosting (AdaBoost): Tree-based boosting methodology where each consecutive base learner assigns
    greater weight to the observations incorrectly predicted by the preceding learner.

    - Gradient Boosting: A boosting methodology where each base learner in the sequence is built to predict the
    residual errors of the model that preceded it.

    - Gradient Boosting Machines (GBMs): Model ensembles that use gradient boosting (they're often called
    black-box models, this is, a model whose predictions cannot be precisely explained).

    - XGBoost: Extreme gradient boosting, an optimized GBM package.






    
## *Exemplar: Course 6 Automatidata project exemplar lab. --> Analyze.*
## *Next: Reference Guide: Validation and Cross Validation. / Exemplar: Build a random forest model. / Case Study.*







## *Pending*

  - *We are using the same data to tune the hyperparameters as we are using to perform model selection. This risks potentially overfitting the
    model to the validation data.*

  - The summary of "More about evaluation metrics for classification models.

  - Unsupervised (Python).

  - Build a decision tree with Python (the part of GridSearchCV).











    











