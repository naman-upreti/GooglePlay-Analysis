import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from google_play_scraper import app, reviews, Sort
import plotly.graph_objs as go
import numpy as np
from textblob import TextBlob
import time
import re
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split

# --- Streamlit UI Config ---
st.set_page_config(page_title="Google Play Store Analysis", layout="wide")

# Custom CSS for styling - Dark/AI Theme
st.markdown("""
<style>
    :root {
        --primary: #00d4ff;
        --secondary: #6c757d;
        --dark: #0a192f;
        --darker: #020c1b;
        --light: #f8f9fa;
        --success: #00d4ff;
        --info: #17a2b8;
        --warning: #ffc107;
        --danger: #dc3545;
    }
    
    .main {
        background-color: var(--darker);
        color: var(--light);
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 100%);
        color: var(--light);
        max-width: 1200px;
        margin: 0 auto;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--primary);
        font-family: 'Segoe UI', Roboto, sans-serif;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
    }
    
    h1 {
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #00d4ff, #090979);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: rgba(10, 25, 47, 0.8);
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: var(--light);
        border-radius: 6px;
        padding: 8px 16px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00d4ff, #090979);
        color: white;
        border: 1px solid var(--primary);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff, #090979);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
    }
    
    .stExpander {
        background-color: rgba(10, 25, 47, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .stExpander .streamlit-expanderHeader {
        color: var(--primary);
        font-weight: 600;
    }
    
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select,
    .stTextArea>div>div>textarea {
        background-color: rgba(10, 25, 47, 0.8) !important;
        color: var(--light) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        border-radius: 6px !important;
    }
    
    .stDataFrame {
        background-color: rgba(10, 25, 47, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 8px !important;
    }
    
    .stRadio>div {
        background-color: rgba(10, 25, 47, 0.8);
        border-radius: 8px;
        padding: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .stMetric {
        background-color: rgba(10, 25, 47, 0.8);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stAlert {
        border-radius: 8px;
    }
    
    /* Custom glowing effect for important elements */
    .glow {
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.5);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }
    
    /* Custom code block styling */
    .stCodeBlock {
        border-radius: 8px;
        background-color: rgba(10, 25, 47, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Custom tab content styling */
    .stTabContent {
        background-color: rgba(10, 25, 47, 0.8);
        border-radius: 0 0 8px 8px;
        padding: 1rem;
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-top: none;
    }
</style>
""", unsafe_allow_html=True)

st.title("📱 Google Play Store Analysis")
st.write("Analyze app details, ratings, and user sentiment using live Google Play data and historical dataset.")

# Caching function to avoid redundant API calls
@st.cache_data
def fetch_app_details(package_name):
    """Fetch app details with retry mechanism."""
    for _ in range(3):  # Retry up to 3 times
        try:
            return app(package_name)
        except Exception as e:
            time.sleep(2)  # Wait before retrying
    return None  # Return None if all attempts fail

@st.cache_data
def fetch_reviews(package_name, count=50):
    """Fetch latest app reviews with retry mechanism."""
    for _ in range(3):  # Retry up to 3 times
        try:
            return reviews(package_name, lang='en', country='us', sort=Sort.NEWEST, count=count)[0]
        except Exception as e:
            time.sleep(2)
    return []  # Return empty list if all attempts fail

# Sentiment analysis function using TextBlob
def get_sentiment(text):
    return "Positive" if TextBlob(text).sentiment.polarity > 0 else "Negative" if TextBlob(text).sentiment.polarity < 0 else "Neutral"

# --- Live App Analysis ---
tab1, tab2, tab3, tab4 = st.tabs(["Live App Analysis", "Dataset Analysis", "App Comparison", "ML Predictions"])

with tab1:
    st.subheader("Live App Analysis")
    
    # Predefined popular apps
    popular_apps = {
        "WhatsApp": "com.whatsapp",
        "Instagram": "com.instagram.android",
        "Facebook": "com.facebook.katana",
        "Twitter": "com.twitter.android",
        "TikTok": "com.zhiliaoapp.musically",
        "Spotify": "com.spotify.music",
        "Netflix": "com.netflix.mediaclient",
        "YouTube": "com.google.android.youtube",
        "Telegram": "org.telegram.messenger",
        "LinkedIn": "com.linkedin.android"
    }

    # Multi-select dropdown
    selected_apps = st.multiselect("Select apps to analyze", list(popular_apps.keys()), default=["WhatsApp"])

    if st.button("Analyze Selected Apps"):
        if not selected_apps:
            st.warning("Please select at least one app to analyze.")
        else:
            comparison_data = []
            for app_name in selected_apps:
                package_name = popular_apps[app_name]
                with st.spinner(f"Fetching data for {app_name}..."):
                    app_details = fetch_app_details(package_name)

                    if app_details:
                        comparison_data.append({
                            'App': app_name,
                            'Rating': app_details['score'],
                            'Reviews': app_details['reviews'],
                            'Installs': app_details['installs'],
                            'Size': app_details.get('size', 'Not available'),
                            'Price': app_details['price']
                        })

                        with st.expander(f"📱 {app_name} Details"):
                            st.write(f"**📛 Name:** {app_details['title']}")
                            st.write(f"**🏢 Developer:** {app_details['developer']}")
                            st.write(f"**⭐ Rating:** {app_details['score']} (out of 5)")
                            st.write(f"**📥 Installs:** {app_details['installs']}")
                            st.write(f"**📅 Last Updated:** {app_details['updated']}")
                            st.image(app_details['icon'], width=100)

                            # Fetch reviews
                            reviews_data = fetch_reviews(package_name, count=50)
                            if reviews_data:
                                df_reviews = pd.DataFrame(reviews_data)[["content", "score"]]
                                
                                # Apply sentiment analysis
                                df_reviews["sentiment"] = df_reviews["content"].apply(get_sentiment)
                                
                                # Sentiment Distribution
                                sentiment_counts = df_reviews["sentiment"].value_counts()
                                fig, ax = plt.subplots()
                                colors = ["green", "red", "gray"] if "Neutral" in sentiment_counts.index else ["green", "red"]
                                sentiment_counts.plot(kind="bar", color=colors, ax=ax)
                                ax.set_title(f"Sentiment Analysis - {app_name}")
                                st.pyplot(fig)
                                
                                # Word Cloud for reviews
                                if len(df_reviews) > 0:
                                    st.subheader("Review Word Cloud")
                                    all_reviews = " ".join(df_reviews["content"].dropna())
                                    if all_reviews.strip():
                                        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_reviews)
                                        fig, ax = plt.subplots(figsize=(10, 5))
                                        ax.imshow(wordcloud, interpolation='bilinear')
                                        ax.axis('off')
                                        st.pyplot(fig)
                                
                                # Anomaly Detection for Fake Reviews
                                st.subheader("⚠️ Fake Review Detection")
                                st.write("Using Isolation Forest to detect potentially fake reviews based on patterns")
                                
                                # Convert reviews to numerical features
                                if len(df_reviews) > 10:  # Need enough samples
                                    # Create features from review text and score
                                    df_reviews['text_length'] = df_reviews['content'].apply(lambda x: len(str(x)))
                                    df_reviews['word_count'] = df_reviews['content'].apply(lambda x: len(str(x).split()))
                                    
                                    # Use Isolation Forest for anomaly detection
                                    X = df_reviews[['score', 'text_length', 'word_count']].fillna(0)
                                    clf = IsolationForest(contamination=0.1, random_state=42)  # 10% anomalies
                                    df_reviews['is_fake'] = clf.fit_predict(X)
                                    df_reviews['is_fake'] = df_reviews['is_fake'].map({1: 'Genuine', -1: 'Potentially Fake'})
                                    
                                    # Show flagged reviews
                                    fake_reviews = df_reviews[df_reviews['is_fake'] == 'Potentially Fake']
                                    if not fake_reviews.empty:
                                        st.dataframe(fake_reviews[['content', 'score', 'is_fake']])
                                    else:
                                        st.write("No suspicious reviews detected")
                                else:
                                    st.write("Not enough reviews for anomaly detection")
                    else:
                        st.error(f"Error fetching data for {app_name}.")

            # Comparative Analysis
            if len(comparison_data) > 1:
                st.subheader("📊 Comparative Analysis")
                df_comparison = pd.DataFrame(comparison_data)

                # Rating Comparison
                fig = go.Figure()
                fig.add_bar(x=df_comparison['App'], y=df_comparison['Rating'], name='Rating')
                fig.update_layout(title='App Ratings Comparison', yaxis_title='Rating')
                st.plotly_chart(fig)

                # Install Base Comparison
                fig = go.Figure()
                fig.add_bar(x=df_comparison['App'], y=df_comparison['Installs'], name='Installs')
                fig.update_layout(title='Install Base Comparison', yaxis_title='Number of Installs')
                st.plotly_chart(fig)

# --- Historical Dataset Analysis ---
with tab2:
    st.subheader("Historical Dataset Analysis")
    
    try:
        # Load dataset
        apps_dataset = pd.read_csv('datasets/apps.csv')

        # Data Cleaning
        # Handle 'Size' column
        apps_dataset['Size'] = apps_dataset['Size'].replace('Varies with device', np.nan)
        apps_dataset['Size'] = apps_dataset['Size'].apply(lambda x: str(float(re.sub(r'[^\d.]', '', str(x))) / 1000) if 'k' in str(x) else x)
        
        # Clean numeric columns
        for col in ['Installs', 'Size', 'Price']:
            # Remove non-numeric characters
            apps_dataset[col] = apps_dataset[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            # Convert to numeric
            apps_dataset[col] = pd.to_numeric(apps_dataset[col], errors='coerce')
        
        # Display basic statistics
        st.write("### Dataset Overview")
        st.write(f"Total number of apps: {len(apps_dataset)}")
        st.write(f"Number of categories: {apps_dataset['Category'].nunique()}")
        
        # Category distribution
        st.write("### Category Distribution")
        category_counts = apps_dataset['Category'].value_counts().head(15)  # Top 15 categories
        fig = go.Figure(data=[go.Bar(x=category_counts.index, y=category_counts.values)])
        fig.update_layout(title="Top 15 App Categories", xaxis_tickangle=-45)
        st.plotly_chart(fig)
        
        # Rating distribution
        st.write("### Rating Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        apps_dataset['Rating'].hist(bins=20, ax=ax)
        ax.set_title("App Ratings Distribution")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Number of Apps")
        # Add vertical line for average rating
        avg_rating = apps_dataset['Rating'].mean()
        ax.axvline(x=avg_rating, color='red', linestyle='--')
        ax.text(avg_rating+0.1, ax.get_ylim()[1]*0.9, f'Avg: {avg_rating:.2f}', color='red')
        st.pyplot(fig)
        
        # Price analysis
        st.write("### Price Analysis")
        paid_apps = apps_dataset[apps_dataset['Type'] == 'Paid']
        fig = go.Figure(data=[go.Histogram(x=paid_apps['Price'])])
        fig.update_layout(title="Price Distribution of Paid Apps")
        st.plotly_chart(fig)
        
        # Free vs Paid Apps
        st.write("### Free vs Paid Apps")
        type_counts = apps_dataset['Type'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=type_counts.index, values=type_counts.values, hole=.3)])
        fig.update_layout(title="Free vs Paid Apps Distribution")
        st.plotly_chart(fig)
        
        # Content Rating Analysis
        st.write("### Content Rating Analysis")
        content_rating_counts = apps_dataset['Content Rating'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=content_rating_counts.index, values=content_rating_counts.values)])
        fig.update_layout(title="Content Rating Distribution")
        st.plotly_chart(fig)
        
        # Load user reviews dataset if available
        try:
            reviews_dataset = pd.read_csv('datasets/user_reviews.csv')
            # Merge apps and reviews datasets
            merged_df = pd.merge(apps_dataset, reviews_dataset, on='App', how='inner')
            merged_df = merged_df.dropna(subset=['Sentiment', 'Translated_Review'])
            
            if not merged_df.empty:
                st.write("### Sentiment Analysis from Historical Reviews")
                
                # Sentiment distribution
                sentiment_counts = merged_df['Sentiment'].value_counts()
                fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values)])
                fig.update_layout(title="Review Sentiment Distribution")
                st.plotly_chart(fig)
                
                # Sentiment by app type
                st.write("### Sentiment by App Type (Free vs Paid)")
                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_by_type = pd.crosstab(merged_df['Type'], merged_df['Sentiment'])
                sentiment_by_type.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title("Sentiment Distribution by App Type")
                ax.set_xlabel("App Type")
                ax.set_ylabel("Count")
                st.pyplot(fig)
        except Exception as e:
            st.info("User reviews dataset not loaded or analysis skipped.")
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# --- App Comparison by Categories ---
with tab3:
    st.subheader("App Comparison by Categories")
    
    try:
        # Load dataset if not already loaded
        if 'apps_dataset' not in locals():
            apps_dataset = pd.read_csv('datasets/apps.csv')
            # Clean data as in tab2
            apps_dataset['Size'] = apps_dataset['Size'].replace('Varies with device', np.nan)
            apps_dataset['Size'] = apps_dataset['Size'].apply(lambda x: str(float(re.sub(r'[^\d.]', '', str(x))) / 1000) if 'k' in str(x) else x)
            for col in ['Installs', 'Size', 'Price']:
                apps_dataset[col] = apps_dataset[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                apps_dataset[col] = pd.to_numeric(apps_dataset[col], errors='coerce')
        
        # Select category
        category = st.selectbox("Select App Category", sorted(apps_dataset["Category"].unique()))
        
        # Filter apps by category
        filtered_apps = apps_dataset[apps_dataset["Category"] == category]
        
        if not filtered_apps.empty:
            st.write(f"### Top Apps in {category} Category")
            
            # Create tabs for different sorting criteria
            sort_criteria = st.radio(
                "Sort by:", 
                ["Rating", "Installs", "Reviews"], 
                horizontal=True
            )
            
            # Sort and display leaderboard
            if sort_criteria == "Rating":
                sorted_apps = filtered_apps.sort_values(by="Rating", ascending=False)
            elif sort_criteria == "Installs":
                sorted_apps = filtered_apps.sort_values(by="Installs", ascending=False)
            else:  # Reviews
                sorted_apps = filtered_apps.sort_values(by="Reviews", ascending=False)
            
            # Display top 20 apps
            st.dataframe(
                sorted_apps[["App", "Rating", "Reviews", "Installs", "Type", "Price"]].head(20),
                use_container_width=True
            )
            
            # Visualization of top 10 apps
            top_10 = sorted_apps.head(10)
            
            st.write(f"### Top 10 {category} Apps by {sort_criteria}")
            fig = go.Figure()
            
            if sort_criteria == "Rating":
                fig.add_bar(x=top_10['App'], y=top_10['Rating'], text=top_10['Rating'], textposition='auto')
                fig.update_layout(yaxis_title='Rating')
            elif sort_criteria == "Installs":
                fig.add_bar(x=top_10['App'], y=top_10['Installs'], text=top_10['Installs'], textposition='auto')
                fig.update_layout(yaxis_title='Number of Installs')
            else:  # Reviews
                fig.add_bar(x=top_10['App'], y=top_10['Reviews'], text=top_10['Reviews'], textposition='auto')
                fig.update_layout(yaxis_title='Number of Reviews')
                
            fig.update_layout(
                title=f"Top 10 {category} Apps by {sort_criteria}",
                xaxis_tickangle=-45,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.write("### Category Insights")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Rating", f"{filtered_apps['Rating'].mean():.2f}")
            with col2:
                st.metric("Free Apps", f"{len(filtered_apps[filtered_apps['Type'] == 'Free'])}")
            with col3:
                st.metric("Paid Apps", f"{len(filtered_apps[filtered_apps['Type'] == 'Paid'])}")
        else:
            st.warning(f"No apps found in the {category} category.")
            
    except Exception as e:
        st.error(f"Error in app comparison: {e}")

# --- ML Predictions ---
with tab4:
    st.subheader("Predict App Success Using Machine Learning")
    
    try:
        # Load dataset if not already loaded
        if 'apps_dataset' not in locals():
            apps_dataset = pd.read_csv('datasets/apps.csv')
            # Clean data as in tab2
            apps_dataset['Size'] = apps_dataset['Size'].replace('Varies with device', np.nan)
            apps_dataset['Size'] = apps_dataset['Size'].apply(lambda x: str(float(re.sub(r'[^\d.]', '', str(x))) / 1000) if 'k' in str(x) else x)
            for col in ['Installs', 'Size', 'Price']:
                apps_dataset[col] = apps_dataset[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                apps_dataset[col] = pd.to_numeric(apps_dataset[col], errors='coerce')
        
        st.write("### Train a Model to Predict App Ratings")
        st.write("This model uses app features to predict its potential rating.")
        
        # Feature selection
        apps_for_model = apps_dataset.dropna(subset=['Rating', 'Installs', 'Size', 'Price'])
        
        # Add categorical features
        apps_for_model = pd.get_dummies(apps_for_model, columns=['Category', 'Content Rating'], drop_first=True)
        
        # Select features and target
        features = ['Installs', 'Size', 'Price'] + [col for col in apps_for_model.columns if col.startswith('Category_') or col.startswith('Content Rating_')][:10]  # Limit to avoid too many features
        X = apps_for_model[features]
        y = apps_for_model['Rating']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # Model evaluation
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                st.write(f"Model R² score on training data: {train_score:.4f}")
                st.write(f"Model R² score on test data: {test_score:.4f}")
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("### Feature Importance")
                fig = go.Figure()
                fig.add_bar(x=feature_importance['Feature'], y=feature_importance['Importance'])
                fig.update_layout(title="Feature Importance for App Rating Prediction", xaxis_tickangle=-45)
                st.plotly_chart(fig)
                
                # Predict for a new app
                st.write("### Predict Rating for a New App")
                st.write("Enter details about your app to predict its potential rating:")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    installs = st.number_input("Expected Installs", min_value=1000, max_value=1000000000, value=100000, step=10000)
                with col2:
                    size = st.number_input("App Size (MB)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
                with col3:
                    price = st.number_input("Price ($)", min_value=0.0, max_value=100.0, value=0.0, step=1.0)
                
                # Create a sample for prediction
                sample = pd.DataFrame([[installs, size, price] + [0] * (len(features) - 3)], columns=features)
                
                # Predict
                predicted_rating = model.predict(sample)[0]
                
                # Display prediction with a gauge chart
                st.write(f"### Predicted App Rating: {predicted_rating:.2f} / 5.0")
                
                # Create a gauge chart for the rating
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted_rating,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Rating"},
                    gauge={
                        'axis': {'range': [0, 5]},
                        'bar': {'color': "#1E88E5"},
                        'steps': [
                            {'range': [0, 2.5], 'color': "#EF5350"},
                            {'range': [2.5, 3.5], 'color': "#FFCA28"},
                            {'range': [3.5, 5], 'color': "#66BB6A"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig)
                
                # Recommendations based on prediction
                st.write("### Recommendations")
                if predicted_rating < 3.0:
                    st.warning("Your app might struggle to gain traction. Consider improving its features or reducing its size.")
                elif predicted_rating < 4.0:
                    st.info("Your app has potential but might need some improvements to stand out.")
                else:
                    st.success("Your app has excellent potential for success!")
        else:
            st.info("Click 'Train Model' to start the prediction process.")
            
    except Exception as e:
        st.error(f"Error in ML prediction: {e}")