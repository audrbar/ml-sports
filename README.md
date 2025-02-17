## A Supervised Machine Learning Project for Text Classification 
### Introduction
This project aim is to develop a supervised machine learning pipeline capable of classifying texts into 
predefined categories. The workflow includes data scraping, text preprocessing, vectorization and the implementation 
of multiple machine learning algorithms. The final outcome is a fully functional pipeline that preprocesses data, 
trains at least three different models, evaluates their performance and determines the best-performing approach 
for text classification.
### Utilities
Utility functions are used throughout the project to adjust pandas display settings, write data to CSV files, 
binary data - to pickle files, load the data from these files, split the datasets into training and test sets, 
also into training, validation and test sets, find and save unique values.
### Data used
A sports articles dataset is used for the project setting, sourced from Sports Illustrated, each labeled with 
their respective categories (e.g., basketball, soccer, football). Articles are scraped using the Python libraries 
`requests` and `BeautifulSoup`, extracting relevant text (title, content) and metadata (category). The structured 
dataset is saved in CSV format, containing:
- `text`: the article content;
- `category`: the predefined category assigned to the article.
> :memo: **Info:** 2400 articles where scraped and saved for processing.

The main dataset / project features, witch are important for the classifiers choosing, are:\
`General text classification` `Small dataset` `Predefined labels` `Multi-class classification` 
`Structured text features` `Low compute`
### Data Preprocessing
The scraped data are prepared for model training in three main steps:
- a missing values are handled, duplicates removed, text columns cleaned, category column extracted, 
categories distribution balanced with `Pandas` library;
- the text is cleaned with `Natural Language Toolkit (NLTK)` (removes HTML tags, punctuation, stopwords, etc.);
- the text is normalized with `NLTK's WordNetLemmatizer` (converts to lowercase, lemmatize (converts words to their 
base form) and `NLTK's PorterStemmer` (stemming reduces words to their root form);
- the text is tokenized and vectorized using **4** methods: `TF-IDF`, `NGRAM`, `Word2Vector`, `FastText` (embeddings).
> :memo: **Info:** 2100 records remaining after handling missing values, removed duplicates.\
> :memo: **Info:** 1270 records remaining after filtering, sampling and some categories dropped.

## Picking Classifiers for Text Classification
Text classification is a fundamental Natural Language Processing (NLP) task that involves assigning predefined 
labels to textual data. Below is a breakdown of different classifiers used for text classification, categorized 
by type and use case. This includes (1) Traditional Machine Learning Classifiers, (2) Deep Learning-Based Classifiers 
(NN's), (3) Pre-Trained Transformer Models.
### 1️⃣ Traditional Machine Learning Classifiers
Traditional ML-based classifiers require **feature engineering** (e.g., TF-IDF, word embeddings) before classification.

| Classifier                            | Best For                                | Pros                                   | Cons                                    | Model Provider        |
|---------------------------------------|-----------------------------------------|----------------------------------------|-----------------------------------------|-----------------------|
| Logistic Regression                   | Binary & multi-class classification     | Simple, efficient, interpretable       | Limited to linear decision boundaries   | `scikit-learn`        |
| Support Vector Machine (SVM)          | Spam detection, sentiment analysis      | Works well for small datasets          | Computationally expensive on large data | `scikit-learn`        |
| Naive Bayes (NB)                      | Email filtering, topic categorization   | Fast, handles small datasets well      | Assumes feature independence            | `scikit-learn`        |
| Random Forest                         | General text classification             | Handles high-dimensional data well     | Slower for large datasets               | `scikit-learn`        |
| Gradient Boosting (XGBoost, LightGBM) | Large-scale classification              | High accuracy, handles imbalanced data | Requires careful tuning                 | `XGBoost`, `LightGBM` |
| k-Nearest Neighbors (k-NN)            | Small datasets, language classification | Simple, non-parametric                 | Slow for large datasets                 | `scikit-learn`        |

**Best For:** Small-to-medium datasets with structured text features (TF-IDF, word embeddings).  
**Libraries:** `scikit-learn`, `XGBoost`, `LightGBM`  

### 2️⃣ Deep Learning-Based Classifiers (Neural Networks)
Deep learning models **learn text representations automatically**, requiring **less feature engineering**.

| Classifier                           | Best For                                      | Pros                               | Cons                                 | Model Provider                   |
|--------------------------------------|-----------------------------------------------|------------------------------------|--------------------------------------|----------------------------------|
| Multilayer Perceptron (MLP)          | General text classification                   | Works well with dense embeddings   | Requires feature engineering         | `TensorFlow`, `Keras`, `PyTorch` |
| Convolutional Neural Networks (CNNs) | Short text classification, sentiment analysis | Captures local patterns in text    | Less effective for long documents    | `TensorFlow`, `Keras`, `PyTorch` |
| Recurrent Neural Networks (RNNs)     | Sequential text classification                | Handles sequential dependencies    | Slower training, vanishing gradients | `TensorFlow`, `Keras`, `PyTorch` |
| LSTMs (Long Short-Term Memory)       | Long text classification, sentiment analysis  | Preserves long-range dependencies  | Computationally expensive            | `TensorFlow`, `Keras`, `PyTorch` |
| GRUs (Gated Recurrent Units)         | Faster alternative to LSTMs                   | Memory efficient                   | Still slower than CNNs               | `TensorFlow`, `Keras`, `PyTorch` |
| Transformers (BERT, RoBERTa, T5)     | Large-scale classification, contextual text   | Best for complex NLP tasks         | Requires GPUs, expensive training    | `Hugging Face Transformers`      |

**Best For:** **Large-scale text classification** with deep contextual understanding.  
**Libraries:** `TensorFlow`, `PyTorch`, `Keras`, `transformers`  

### 3️⃣ Pre-Trained Transformer Models (State-of-the-Art)
Pre-trained **transformer models** have revolutionized NLP, offering state-of-the-art accuracy for text classification.

| Model                       | Provider     | Best For                                             | Pros                                                    | Cons                      |
|-----------------------------|--------------|------------------------------------------------------|---------------------------------------------------------|---------------------------|
| BERT                        | Google AI    | Sentiment analysis, topic classification             | Strong contextual understanding, bidirectional learning | Slow inference            |
| DistilBERT                  | Hugging Face | Fast classification                                  | Lighter than BERT, optimized for speed                  | Slight accuracy trade-off |
| RoBERTa                     | Meta AI      | Text classification, fake news detection             | More robust than BERT                                   | Requires fine-tuning      |
| GPT-4                       | OpenAI       | Zero-shot classification                             | No training required, API-based                         | Requires API access       |
| Text-to-Text Transformer T5 | Google AI    | Multi-task learning (classification + summarization) | Flexible for various NLP tasks                          | Large model size          |
| XLNet**                     | Google AI    | Long-form classification                             | Handles dependencies better than BERT                   | Computationally expensive |
| Longformer                  | Hugging Face | Classification of long articles                      | Optimized for processing long documents                 | Requires large datasets   |

**Best For:** **Large datasets & complex classification tasks**  
**Libraries:** `transformers`, `PyTorch`, `TensorFlow`

### 🏆Selected Classifiers
A particular classification algorithm outperforms others on particular dataset depending on dataset's structure, shape, 
density and noise. Selected Classifiers to evaluate in the project:

| Traditional Classifiers    | Deep Learning-Based Classifiers (NN's)    | Pre-Trained Transformer |
|----------------------------|-------------------------------------------|-------------------------|
| Logistic Regression        | Simple Neural Network (SNN)               | DistilBERT              |
| Random Forest              | Sequential Recurrent Neural Network (RNN) |                         |
| Decision Tree              |                                           |                         |
| k-Nearest Neighbors (k-NN) |                                           |                         |

## Traditional Classifiers
![Data Plot](./img/traditional_img.png)
#### Traditional Classifiers Performance evaluation metrics:
![Data Plot](./img/traditional_table.png)
#### KNeighbors Classifier fine-tuning GridSearchCV params:
- `metric` euclidean, manhattan, minkowski;
- `n_neighbors` 3, 5, 7, 10;
- `weights` uniform, distance.
#### KNeighbors Classifier Best Params Across Different Vectorization Methods:
| Dataset  |     Metric | n_neighbors |     weights | Mean Cross-Validation Accuracy |
|----------|-----------:|------------:|------------:|-------------------------------:|
| TF-IDF   |  euclidean |           7 |    distance |                         0.9056 |
| Word2Vec |  euclidean |          10 |    distance |                         0.7757 |
| N-Gram   |  euclidean |          10 |    distance |                         0.8888 |
| FastText |  euclidean |           7 |    distance |                         0.8515 |
#### KNeighbors Classifier Performance Comparison Across Different Vectorization Methods:
![Data Plot](./img/knn_performance.png)
![Data Plot](./img/knn_table.png)
#### KNeighbors Classifier Confusion Matrix on N-Gram Vectorization:
![Data Plot](./img/knn_matrix.png)
### Logistic Regression Classifier fine-tuning GridSearchCV params:
- `threshold` 0.01, 0.1, 1, 10;
- `solver` lbfgs, liblinear, saga;
- `max_iter` 100, 200, 300, 600.
#### Logistic Regression Classifier Best Params Across Different Vectorization Methods:
| Dataset  | Threshold | Max_Iter | Solver |  Accuracy |
|----------|----------:|---------:|-------:|----------:|
| TF-IDF   |        10 |      100 |   saga |    0.9412 |
| Word2Vec |        10 |      200 |  lbfgs |    0.8745 |
| N-Gram   |        10 |      100 |   saga |    0.9255 |
| FastText |        10 |      200 |  lbfgs |    0.8706 |
#### Logistic Regression Classifier Best Params Across Different Vectorization Methods:
![Data Plot](./img/lr_performance.png)
![Data Plot](./img/lr_table.png)
#### Logistic Regression Classifier Confusion Matrix on TF-IDF Vectorization:
![Data Plot](./img/lr_matrix.png)
## Neural Network Models
### Simple Neural Network (SNN) - Tensorflow Keras Sequential Model
It is a simple (basic) fead-forward neural network model. Input layer accepts vectorized text. Hidden layers fully 
connected with activation functions (e.g., ReLU). Output layer softmax for multi-class classification.
![Data Plot](./img/snn_model.png)
#### Data Shape used for SNN tuning:
![Data Plot](./img/snn_data_shape.png)
#### Hyperparameters used for tuning:
- `neurons_list` [512, 256], [1024, 512], [256, 128];
- `dropout_rates_list` [0.3, 0.3], [0.5, 0.5], [0.2, 0.2];
- `batch_sizes` 32, 64;
- `epochs` 100.
![Data Plot](./img/snn_tuning_results.png)
![Data Plot](./img/snn_best_params.png)
![Data Plot](./img/snn_test_accuracy.png)
### Recurrent Neural Network (RNN) - Long Short-Term Memory (LSTM)
RNN variants are LSTM or GRU designed for handling sequential text data. 
![Data Plot](./img/rnn_model.png)
#### FastText dataset results
![Data Plot](./img/rnn_best_model.png)
### Transformer-based models use BERT, RoBERTa, or similar pre-trained models.
Distil Bert Model from pretrained distilbert-base-uncased model Leverages advanced deep learning techniques 
for better performance.
![Data Plot](./img/bert_dataset.png)
![Data Plot](./img/bert_results.png)
## Performance Comparison
While deep learning models like DistilBERT offer contextual understanding, traditional ML models (Logistic Regression) 
with TF-IDF delivered similar accuracy with significantly lower computational cost. This suggests that for structured 
text datasets, feature engineering remains a powerful tool and deep learning models should be carefully fine-tuned 
to justify their resource requirements.

| Classifier                              |          Vectorizer | Model Size | Accuracy |
|-----------------------------------------|--------------------:|-----------:|---------:|
| Logistic Regression                     |              TF-IDF |       8 KB |   0.9412 |
| k-Nearest Neighbors                     |              N-Gram |     416 KB |   0.9059 |
| Tensorflow Keras Sequential Model (SNN) |              TF-IDF |    32,4 MB |   0.9412 |
| Distil Bert Base Uncased Model (RNN)    | DistilBertTokenizer |   267,6 MB |   0.8745 |
# 🏆 Models Winners in Classifying Articles are Logistic Regression and Tensorflow Keras Sequential Model (SNN)
## Conclusions
This project aimed to classify text data using various machine learning and deep learning models, leveraging different 
vectorization techniques. We evaluated models ranging from traditional ML classifiers (Logistic Regression, k-NN) 
to deep learning architectures (SNN and RNN-based DistilBERT). The key finding was that Logistic Regression with 
TF-IDF and SNN with TF-IDF performed best, both achieving 94.12% accuracy, demonstrating that traditional models 
can be competitive with deep learning when using appropriate feature engineering. However, DistilBERT (RNN) showed 
lower accuracy (87.45%), suggesting that fine-tuning on this dataset might improve performance.
#### Challenges and Solutions
- `Data Imbalance`: Some categories had fewer samples, affecting training balance. This was addressed by applying 
category filtering and sampling to ensure a more uniform class distribution.
- `Model Convergence Issues`: The lbfgs solver in Logistic Regression faced convergence warnings, which were mitigated 
by scaling data and increasing max iterations.
- `Computational Constraints`: Deep learning models, particularly DistilBERT, had large model sizes (267.6 MB) 
and longer inference times. Optimizations such as reducing sequence length and batch size helped improve efficiency.
- `Deprecation Warnings in Pandas`: Deprecated behavior in groupby().apply() was updated using group_keys=False 
and include_groups=False.
#### Future Scope
Potential improvements (e.g., using more data, fine-tuning models, or adding categories).

