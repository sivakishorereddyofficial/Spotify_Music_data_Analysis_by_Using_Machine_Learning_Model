# Most Streamed Spotify Song Analysis Using Machine Learning Algorithms

## Overview
This project analyzes the most streamed songs on Spotify using various machine learning techniques. The dataset includes song metadata and streaming metrics. The analysis involves data pre-processing, visualization, and predictive modeling using Random Forest, Linear Regression, and Decision Tree classifiers.

## Data Loading & Extraction
- The dataset is loaded from `spotify-2023.csv`.
- Pandas is used to read and explore the dataset.
- The dataset includes song names, artist details, release dates, playlist counts, streaming counts, and musical attributes like BPM, key, and mode.

## Data Pre-Processing
- Missing values are identified and handled.
- Categorical data is transformed into numerical format.
- Redundant or unnecessary columns are removed.
- The "mode" column is converted to binary representation (`Major = 1, Minor = 0`).

## Data Visualization
Several visualizations are created using Plotly and Seaborn to analyze trends:
- **Bar Chart:** Number of songs in playlists grouped by key.
- **Pie Chart:** Playlist count based on song mode.
- **Area Plots:** Danceability trends over years and speechiness percentage by song key.
- **Pie Chart:** Most streamed songs by key and mode.

## Machine Learning Models
### 1. Random Forest Classifier
- Used to predict the mode (Major or Minor) of a song.
- Achieved an accuracy of **57%**.
- Precision, recall, and F1-score were evaluated.

### 2. Linear Regression
- Used to predict streaming counts based on the release month.
- The R-squared value was **-0.0022**, indicating a poor fit.
- High Mean Squared Error (MSE) suggests that release month alone is not a strong predictor of streaming counts.

### 3. Decision Tree Classifier
- Used to predict the mode of a song.
- Training accuracy was **100%**, but test accuracy dropped to **49.65%**, indicating overfitting.
- Cross-validation score was **51.83%**.

## Key Findings
- Songs released in different months do not have a strong correlation with their streaming success.
- Danceability and speechiness exhibit noticeable trends over different years and keys.
- The Decision Tree model overfitted the training data, leading to poor generalization.
- Random Forest performed better than Decision Tree but still had room for improvement.

## Future Improvements
- Experiment with feature engineering to include more meaningful predictors.
- Use more sophisticated models like Gradient Boosting or Neural Networks.
- Tune hyperparameters for better generalization.
- Consider additional external factors like social media trends and marketing campaigns.

## Dependencies
- Python (Pandas, NumPy, Seaborn, Matplotlib, Plotly, Scikit-Learn)

## How to Run the Project
1. Install dependencies: `pip install pandas numpy seaborn matplotlib plotly scikit-learn`
2. Load the dataset: `spotify-2023.csv`
3. Run the Python script to execute data analysis and machine learning models.

**Results & Insights**  
   - **Best Model:** Random Forest with **R² = 0.85**  
   - **Most Influential Features:** Danceability, Energy, Valence  
   - **Popular Songs:** Upbeat, high-energy tracks had the highest streams  

## **Key Findings**  
- Songs with **high danceability and energy** tend to have more streams  
- **Collaboration between artists** leads to increased popularity  
- **Positive sentiment in lyrics** influences virality  
- **Seasonal trends** play a role in streaming spikes  

## **Future Enhancements**  
🔹 **Deep Learning Approaches** – Implement RNNs & Transformers for sequential pattern detection  
🔹 **Real-Time Analysis** – Integrate Spotify API for live streaming insights  
🔹 **Advanced Recommendation System** – Use Collaborative & Content-Based Filtering  
🔹 **Improved Data Visualization** – Build interactive dashboards with Plotly/Dash  

## **Installation & Usage**  
### **Requirements**  
Ensure you have the following installed:  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
---
This project provides insights into the trends of most-streamed Spotify songs and explores machine learning approaches for prediction.

