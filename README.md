# 📱 Google Play Store Analytics Dashboard

![Dashboard Screenshot](images/googleplayanalysis.png)

An AI-powered analytics platform for deep insights into Google Play Store apps, leveraging real-time data, machine learning, and interactive visualizations.

---

## 🌟 Key Features

### 🔍 Live App Analysis
| Feature | Description | Technology |
|---------|-------------|------------|
| ✨ Real-time Data Fetching | Instantly fetch app metrics from Google Play | google-play-scraper |
| 📊 Sentiment Analysis | Categorize reviews as Positive/Negative/Neutral | TextBlob |
| 🔒 Fake Review Detection | Identify suspicious reviews using ML | Isolation Forest |
| 💡 Visual Analytics | Interactive charts & word clouds | WordCloud, Plotly |

### 📊 Historical Trends & Insights
- 🌐 **Category Distribution:** Market share across genres
- ⏳ **Rating Patterns:** Evolution of ratings over time
- 💸 **Price Sensitivity:** Free vs. paid app performance
- 🎉 **Content Ratings:** Audience segmentation
- 💡 **Feature Benchmarking:** Key attributes for success

### ↔️ Comparative Analysis
```mermaid
graph TD
    A[Select Category] --> B[Filter Apps]
    B --> C[Sort by Metric]
    C --> D[Visual Comparison]
    D --> E[Insights]
```

### 🤖 ML-Powered Predictions
1. 🎯 **Success Probability Estimator**
2. 🔢 **Future Rating Predictor**
3. 🔍 **Feature Importance Analysis**
4. ⚖️ **Automated Recommendations**

---

## 🛠️ Technology Stack

### **Frontend:**
- 🛠️ **Streamlit** (Python Web Framework)
- 🌟 **Custom Dark/AI Theme** (CSS)
- 📊 **Plotly & Matplotlib** (Data Visualization)

### **Backend & Processing:**
```python
# Core Processing
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest

# NLP & Text Analysis
from textblob import TextBlob
from wordcloud import WordCloud

# Data Collection
from google_play_scraper import app, reviews
```

### **Data Pipeline:**
```plaintext
Live Data ➔
Data Cleaning ➔
Feature Engineering ➔
Analysis ➔
Visualization ➔
Insights
```

---

## 🚀 Installation & Usage

```bash
# Clone repository
git clone https://github.com/naman-upreti/GooglePlay-Analysis.git
cd GooglePlay-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### **System Requirements:**
- Python 3.8+
- 8GB RAM (for ML components)
- Stable internet connection

---

## 📊 Sample Outputs

### **Sentiment Analysis**
![Sentiment Chart](https://via.placeholder.com/400x300.png?text=Sentiment+Analysis)

### **Rating Prediction Output**
```python
{
    "Predicted_Rating": 4.2,
    "Confidence": 0.85,
    "Key_Factors": ["App Size", "Category", "Price"]
}
```

---

## 🎯 Business Applications

| Industry | Use Case |
|----------|----------|
| 💻 App Development | Feature prioritization |
| 📊 Marketing | Review sentiment tracking |
| 💼 Product Management | Competitor benchmarking |
| 💰 Investment | Market trend analysis |

---

## 📍 Upcoming Features
- [ ] ⏳ **User Review Timeline Analysis**
- [ ] 🔄 **Cross-Platform Comparison (iOS vs. Android)**
- [ ] 🔀 **Advanced Review Clustering**
- [ ] 🎉 **Automated Report Generation**

---

## 👨‍💻 Developer

**Naman Upreti**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/naman-upreti)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-lightgrey)](https://github.com/naman-upreti)

---

## 🔒 License
MIT License  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Contribute to the Project

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

**Bug Reports & Feature Requests:**  
Submit issues via [GitHub Issues](https://github.com/naman-upreti/GooglePlay-Analysis/issues)

---

<div align="center">
  <sub>Built with ❤️ by Naman | Last Updated: April 2025</sub>
</div>

