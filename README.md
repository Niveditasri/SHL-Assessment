# SHL-Assessment
## Grammar Scoring Engine - Final Report

### Objective
To build a regression model that predicts grammar scores (0–5 scale) from spoken audio files, using Wav2Vec2 embeddings and deep learning models.

### Dataset
- 444 training samples and 195 test samples of spoken audio in `.wav` format.
- Ground truth labels are MOS Likert-based grammar scores ranging from 1 to 5.

### Preprocessing
- Audio files were resampled to 16kHz.
- Features extracted using Wav2Vec2.
- Missing files handled with zero vectors.
- MinMaxScaler applied for scaling.
- PCA applied to reduce feature dimensionality where needed.
- CNN+LSTM input reshaped into 3D.

### Models Used
- **Random Forest Regressor**
- **XGBoost Regressor**
- **CNN + LSTM** model
- **Wav2Vec2 + Linear Regression**
- **Meta-Learner (Linear Regression)** combining all above

### Ensemble Techniques
- Simple Averaging
- Weighted Averaging (based on validation performance)
- Meta-Learner with Linear Regression

### Evaluation Metrics
- **Pearson Correlation** (Primary metric)
- **MAE**, **MSE**, **R² Score**
- Visualizations: True vs Predicted scatter plots with regression line

### Results Summary
- Best performance achieved using **Meta-Learner Ensemble**
- Visualizations showed strong alignment with actual labels
- Feature extraction via Wav2Vec2 proved highly effective

### Conclusion
This project demonstrates the effectiveness of audio feature extraction combined with ensemble learning for grammar scoring. Further improvement can be made with larger datasets or transformer-based fine-tuned models.
