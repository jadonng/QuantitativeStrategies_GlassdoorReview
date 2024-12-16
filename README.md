# Quantitative Strategies with Glassdoor Reviews

A repository owned by `Group 3` for FINA4359: Data Analytics, Quantitative Finance, and Blockchain Finance by the University of Hong Kong. The objective was to conduct alpha-signal research based on a textual employer ratings data set.
This project was supervised by Dr. Yang You and his research team. The final paper can be found [here](https://theoobadiahteguh.net/assets/glassdoor.pdf).
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
