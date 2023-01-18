# Media Prediction

This project is to estimate the maximum predictability of music listening and web browsing trajectories. 

And built a Markov Chain model for the prediction.

It includes evaluation of the Markov Chain model - Accuracy of markov chain prediction, and a Leave-One-Out Cross Validation accuracy evaluation. 

## Data
1. Last.fm Music Recommendation Dataset (1K users).
2. Web History Repository Dataset (~500 users)

## Code
code is in utils directory
1. Entropy.py contains functions that can evaluate the emprical entropy of the empirical sequences.
2. Prediction.py contains functions that can evaluate the maximum predictability of the empirical sequences.
3. Markov.py contains functions that can build a Markov Chain model and estimate its Leave-One-Out prediction accuracy. 

## Analysis
Analysis is conducted as jupyter notebooks. 
1. music_listening_analysis conduct analysis for music listening trajectories.
2. web_browsing_analysis conduct analysis for web browsing trajectories.
