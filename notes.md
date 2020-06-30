### Data Preparation Packages
- DataFrames is like pandas
- ExcelReaders to read .xlsx documents
- CategoricalArrays is a package for handling categorical data, 
- CSV is the package for handling tabular data in csv format


### Libraries for Data Models
- Unsupervised Learning
    - Clustering
    - ManifoldLearning
- Supervised Learning (transparent models)
    - GLM - Generalized Linear Model
    - Decision Tree
- Supervised Learning (black box models)
    - XGBoost
    - LibSVM
- MLJ and MLJModels are two packages ranging from kNN to Ridge Regresssors, to emnsables.

### Statistics Libraries
- StatsBase - mature stats library
- Distributions - pdf and logpdf
- HypothesisTests - t-tests, chi-squared, and more hypothesis testing
- MultivariateStats - Principle Component Analysis

### Utility Libraries
- Distances
- MLJ - grid search, k-fold cross validation, and more
- ScikitLearn
- MLLablelUtils - label encoding
- MLBase
- AutoMLPipeline - AMLP - https://github.com/IBM/AutoMLPipeline.jl

### AI Libraries
- Knet  
- Flux
- Metalhead

### Other libraries
- TSne - T-SNE algorithm
- UMAP
- Graphs
- Gadfly
- PyPlot
- Pkg

### Examples of use
- `Pkg.add(CoolPackage)`
- `Pkg.update()`
- `Pkg.rm(CoolPackage)`
- `Pkg.status()`

### REPL Pkg usage
- From the REPL, `]` goes into Package mode.
- From `]`, the `backspace` key goes back into the REPL


## Two Main Data Analytics Paradigms
- Data analytics is the super-set of data-science and the precursor to data-mining
- Paradigm #1 - Conventional statistical models make use of models related to distributions to understand the data and make predictions. This is called the "model-driven" approach. 
- Paradigm #2 - Non-parametric Statistics, which means to analyze the data without using distribution based models. This relies on the data itself rather than assumptions of distributions. This is called the "data-driven" approach
- This isn't quite the same distinction that Geron makes in http://homl.info/ as the difference between model-based learning and instance-based learning.. But it may be valuable to compare these two paradigms with Geron's instance/model distinction.
- Hybrid approaches exist that use both paradigms. One such hybrid approach is Bayesian Statistics.
- Pursuing one paradigm while neglecting the other is sophomoric folly.
- ML makes less assumptions about the data


## Terms
- Predictive analytics is data analytics focused on predictions. This is called inferencial modeling.
- AI != ML != Data Analytics != Data Science
- AI is usually the whole system while an ML model is a specific component


## Types of Learning
- Supervised learning
    - KNN
    - Decision trees
    - SVMs
    - Random Forests
    - Boosted Forests
    - Artificial Neural Networks and multilevel perceptrons
- Unsupervised learning
    - dimensionality reduction
    - clustering
    - distinct and independent but can be used together since clustering works better w/ fewer variables involved
- Reinforcement learning
- Semi-supervised learning
- Self-supervised learning (ideal w/ limited labeled data)
- Recommendation systems

## ML for Business
- ML is a means to an end or a business resource
- Stakeholders care about:
    - The effect on the bottom line
    - Return on investment
    - How likely ML is to be an asset
    - The resources required to make this happen
    - The risk of this whole process. How likely is this to become a liability?


