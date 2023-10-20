# Media Prediction

This project is to estimate the predictability of media selection in a static movie selection task,
and estimate the maximum predictability of music listening and web browsing trajectories. 

It builds linear logistic model, SVM model, and Markov Chain Model to predict media selections.

It includes evaluation of models - Accuracy of prediction, and a Leave-One-Out Cross Validation accuracy evaluation. 


## Data
1. Last.fm Music Recommendation Dataset (1K users).
2. Web History Repository Dataset (~500 users)
3. Movie Selection Dataset from [Gong et al. 2023](https://doi.org/10.1093/joc/jqad020)

## Code
code is in utils directory
1. Entropy.py contains functions that can evaluate the emprical entropy of the empirical sequences.
2. Prediction.py contains functions that can evaluate the maximum predictability of the empirical sequences.
3. Markov.py contains functions that can build a Markov Chain model and estimate its Leave-One-Out prediction accuracy. 

## Analysis
Analysis is conducted as jupyter notebooks. 
1. music_listening_analysis conduct analysis for music listening trajectories.
2. web_browsing_analysis conduct analysis for web browsing trajectories.
3. movie_selection_data_split.ipynb splits the movie selection dataset into train and test dataset
4. svm_model_fitting.ipynb conduct SVM predictive model fitting for movie selection dataset
5. llm_model_fitting_comparison.R conduct linear logistic predictive modeling fitting for movie selection dataset
