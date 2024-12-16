# Quantitative Strategies with Glassdoor Reviews

A repository owned by `Group 3` for FINA4359: Data Analytics, Quantitative Finance, and Blockchain Finance by the University of Hong Kong. We conducted alpha-signal research based on a textual employer ratings data set.
The project was supervised by Dr. Yang You and his team. The final research paper can be found [here](https://theoobadiahteguh.net/assets/glassdoor.pdf).
## Our Group Members and UIDs

- (UID: 3036076067) Jadon Ng Tsz Hei
- (UID: 3035756751) Rhenald Louwos
- (UID: 3036077774) Poon Tsz Chung
- (UID: 3035898872) Theo Obadiah Teguh

## Model Description and Computing Resources

On top of the numerical feature-based alpha signals, we performed topic modelling utilizing Singular Value Decomposition (SVD) for preprocessing reviews, coupled with Latent Dirichlet Allocation (LDA), with BERT-Twitter by Barbieri et al. (2020) from the Hugging Face repository. We used the following cloud computing resources to generate sentiment scores with the pre-trained model.

- NVIDIA RTX 2080Ti (~100 compute hours) provided by the HKU School of Computing and Data Science.
- Google Colab (~50 compute hours) free tier facilities.

## Directory Structure
We have separated the source code based on the following structure. The folder `analysis` contains several notebooks we used when performing alpha research. The `backtesting` directory contains our backtesting system as well as several notebooks for testing our generated signals. Then, `eda` contains some notebooks for Exploratory Data Analysis (EDA) and `preprocessing` contains a collection of notebooks for textual data preprocessing. Finally, we have `utils` which contains some helper functions and a `requirements.txt` file to list all our dependencies.

```bash
├── README.md
├── analysis
│   ├── hold_PF.ipynb
│   ├── ratingleadership.ipynb
│   └── regression.ipynb
├── backtesting
│   ├── Backtester.py
│   ├── backtest_draft.ipynb
│   └── backtest_strategies.ipynb
├── eda
│   ├── EDA.ipynb
│   └── EDA_aggdf.ipynb
├── preprocessing
│   ├── preprocess_OLS_df.ipynb
│   ├── preprocess_feature_db.ipynb
│   ├── preprocess_ratings.ipynb
│   ├── preprocess_returns.ipynb
│   ├── preprocess_toDataframes.ipynb
│   ├── preprocess_topic-sentiment-prep.ipynb
│   └── preprocess_topic-sentiment.ipynb
├── requirements.txt
└── utils
    ├── cleaning.ipynb
    └── helper.py
```
