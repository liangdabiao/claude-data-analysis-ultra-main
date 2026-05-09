"""
Sentiment Analysis Tools

This module provides comprehensive sentiment analysis capabilities using both
traditional NLP methods (VADER) and LLM-enhanced analysis for nuanced emotion detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import warnings

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis toolkit.

    Provides traditional VADER sentiment analysis and preparation
    for LLM-enhanced sentiment analysis.
    """

    def __init__(self):
        """Initialize the SentimentAnalyzer."""
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()

        # Sentiment labels
        self.sentiment_labels = {
            'positive': 'Positive',
            'negative': 'Negative',
            'neutral': 'Neutral'
        }

        # Extended sentiment labels for LLM analysis
        self.extended_labels = {
            'very_positive': 'Very Positive',
            'positive': 'Positive',
            'neutral': 'Neutral',
            'negative': 'Negative',
            'very_negative': 'Very Negative'
        }

    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER (Valence Aware Dictionary and sEntiment Reasoner).

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'label': 'Neutral'
            }

        # Get VADER scores
        scores = self.vader_analyzer.polarity_scores(text)

        # Determine sentiment label based on compound score
        if scores['compound'] >= 0.05:
            label = 'Positive'
        elif scores['compound'] <= -0.05:
            label = 'Negative'
        else:
            label = 'Neutral'

        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'label': label
        }

    def analyze_batch_vader(self, texts: List[str]) -> List[Dict[str, Union[float, str]]]:
        """
        Analyze sentiment for a batch of texts using VADER.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment analysis results
        """
        results = []
        for text in texts:
            result = self.analyze_sentiment_vader(text)
            result['text'] = text
            results.append(result)

        return results

    def classify_sentiment_distribution(self, texts: List[str]) -> Dict[str, Union[int, float]]:
        """
        Classify sentiment distribution across a collection of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with sentiment distribution statistics
        """
        if not texts:
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}

        # Analyze all texts
        results = self.analyze_batch_vader(texts)

        # Count sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for result in results:
            label = result['label'].lower()
            sentiment_counts[label] += 1

        # Calculate percentages
        total = len(texts)
        sentiment_percentages = {
            f'{sentiment}_count': count for sentiment, count in sentiment_counts.items()
        }
        sentiment_percentages.update({
            f'{sentiment}_percent': (count / total) * 100
            for sentiment, count in sentiment_counts.items()
        })
        sentiment_percentages['total'] = total

        return sentiment_percentages

    def get_emotion_words(self, texts: List[str],
                         sentiment_type: str = 'positive',
                         top_k: int = 20) -> List[Tuple[str, int]]:
        """
        Extract most frequent words associated with a specific sentiment.

        Args:
            texts: List of texts to analyze
            sentiment_type: Type of sentiment ('positive', 'negative', 'neutral')
            top_k: Number of top words to return

        Returns:
            List of (word, frequency) tuples
        """
        from scripts.text_analyzer import TextAnalyzer

        if not texts:
            return []

        # Initialize text analyzer
        text_analyzer = TextAnalyzer()

        # Analyze sentiment for each text
        sentiment_texts = []
        for text in texts:
            result = self.analyze_sentiment_vader(text)
            if result['label'].lower() == sentiment_type.lower():
                sentiment_texts.append(text)

        if not sentiment_texts:
            return []

        # Tokenize and count words
        all_words = []
        for text in sentiment_texts:
            tokens = text_analyzer.tokenize_text(text)
            all_words.extend(tokens)

        # Count word frequencies
        word_freq = {}
        for word in all_words:
            if len(word) > 2:  # Filter out very short words
                word_freq[word] = word_freq.get(word, 0) + 1

        # Return top k words
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def create_sentiment_wordcloud(self, texts: List[str],
                                   sentiment_type: str = 'positive',
                                   width: int = 800,
                                   height: int = 400) -> WordCloud:
        """
        Create a word cloud for texts with specific sentiment.

        Args:
            texts: List of texts to analyze
            sentiment_type: Type of sentiment to visualize
            width: Word cloud width
            height: Word cloud height

        Returns:
            WordCloud object
        """
        # Get emotion words
        emotion_words = self.get_emotion_words(texts, sentiment_type)

        if not emotion_words:
            return None

        # Create word frequency dictionary
        word_freq = dict(emotion_words)

        # Create word cloud
        if sentiment_type == 'positive':
            colormap = 'Greens'
        elif sentiment_type == 'negative':
            colormap = 'Reds'
        else:
            colormap = 'Blues'

        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            colormap=colormap,
            max_words=100,
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(word_freq)

        return wordcloud

    def plot_sentiment_distribution(self, texts: List[str],
                                   title: str = "Sentiment Distribution",
                                   figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot sentiment distribution for a collection of texts.

        Args:
            texts: List of texts to analyze
            title: Plot title
            figsize: Figure size tuple

        Returns:
            Matplotlib figure object
        """
        # Get sentiment distribution
        distribution = self.classify_sentiment_distribution(texts)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar chart
        sentiments = ['Positive', 'Negative', 'Neutral']
        counts = [distribution.get('positive_count', 0),
                 distribution.get('negative_count', 0),
                 distribution.get('neutral_count', 0)]

        bars = ax1.bar(sentiments, counts, color=['green', 'red', 'gray'])
        ax1.set_title('Sentiment Count Distribution', fontweight='bold')
        ax1.set_ylabel('Number of Texts')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')

        # Pie chart
        percentages = [distribution.get('positive_percent', 0),
                      distribution.get('negative_percent', 0),
                      distribution.get('neutral_percent', 0)]

        colors = ['lightgreen', 'lightcoral', 'lightgray']
        wedges, texts, autotexts = ax2.pie(percentages, labels=sentiments, colors=colors,
                                          autopct='%1.1f%%', startangle=90)

        ax2.set_title('Sentiment Percentage Distribution', fontweight='bold')

        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def analyze_sentiment_trends(self, texts: List[str],
                                timestamps: List[str] = None,
                                time_window: str = 'daily') -> pd.DataFrame:
        """
        Analyze sentiment trends over time.

        Args:
            texts: List of texts to analyze
            timestamps: List of timestamps corresponding to texts
            time_window: Time aggregation window ('daily', 'weekly', 'monthly')

        Returns:
            DataFrame with sentiment trends
        """
        if not texts:
            return pd.DataFrame()

        # Analyze sentiment for all texts
        results = self.analyze_batch_vader(texts)

        # Create DataFrame
        df = pd.DataFrame(results)

        # Add timestamps if provided
        if timestamps:
            df['timestamp'] = pd.to_datetime(timestamps)
            df.set_index('timestamp', inplace=True)

            # Aggregate by time window
            if time_window == 'daily':
                df_resampled = df.resample('D')
            elif time_window == 'weekly':
                df_resampled = df.resample('W')
            elif time_window == 'monthly':
                df_resampled = df.resample('M')
            else:
                df_resampled = df.resample('D')

            # Calculate sentiment statistics
            trends = df_resampled.agg({
                'compound': ['mean', 'count'],
                'positive': 'mean',
                'negative': 'mean',
                'neutral': 'mean'
            }).round(3)

            # Flatten column names
            trends.columns = [f'{col[0]}_{col[1]}' for col in trends.columns]

            return trends.reset_index()

        else:
            # Simple index-based trends
            return df.reset_index(drop=True)

    def compare_sentiment_between_groups(self,
                                        group1_texts: List[str],
                                        group2_texts: List[str],
                                        group1_name: str = "Group 1",
                                        group2_name: str = "Group 2") -> Dict:
        """
        Compare sentiment between two groups of texts.

        Args:
            group1_texts: First group of texts
            group2_texts: Second group of texts
            group1_name: Name for first group
            group2_name: Name for second group

        Returns:
            Dictionary with comparison results
        """
        # Analyze sentiment for both groups
        group1_results = self.analyze_batch_vader(group1_texts)
        group2_results = self.analyze_batch_vader(group2_texts)

        # Calculate statistics
        group1_compound = [r['compound'] for r in group1_results]
        group2_compound = [r['compound'] for r in group2_results]

        comparison = {
            'group1': {
                'name': group1_name,
                'count': len(group1_texts),
                'mean_compound': np.mean(group1_compound),
                'std_compound': np.std(group1_compound),
                'positive_percent': len([r for r in group1_results if r['label'] == 'Positive']) / len(group1_results) * 100,
                'negative_percent': len([r for r in group1_results if r['label'] == 'Negative']) / len(group1_results) * 100,
                'neutral_percent': len([r for r in group1_results if r['label'] == 'Neutral']) / len(group1_results) * 100
            },
            'group2': {
                'name': group2_name,
                'count': len(group2_texts),
                'mean_compound': np.mean(group2_compound),
                'std_compound': np.std(group2_compound),
                'positive_percent': len([r for r in group2_results if r['label'] == 'Positive']) / len(group2_results) * 100,
                'negative_percent': len([r for r in group2_results if r['label'] == 'Negative']) / len(group2_results) * 100,
                'neutral_percent': len([r for r in group2_results if r['label'] == 'Neutral']) / len(group2_results) * 100
            }
        }

        # Statistical test (t-test)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(group1_compound, group2_compound)
        comparison['statistical_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }

        return comparison

    def get_sentiment_summary(self, texts: List[str]) -> Dict[str, Union[str, float, int]]:
        """
        Get a comprehensive sentiment summary for a collection of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            Dictionary with sentiment summary
        """
        if not texts:
            return {}

        # Analyze all texts
        results = self.analyze_batch_vader(texts)

        # Extract scores
        compound_scores = [r['compound'] for r in results]
        positive_scores = [r['positive'] for r in results]
        negative_scores = [r['negative'] for r in results]
        neutral_scores = [r['neutral'] for r in results]

        # Get distribution
        distribution = self.classify_sentiment_distribution(texts)

        # Create summary
        summary = {
            'total_texts': len(texts),
            'average_compound_score': np.mean(compound_scores),
            'median_compound_score': np.median(compound_scores),
            'std_compound_score': np.std(compound_scores),
            'min_compound_score': np.min(compound_scores),
            'max_compound_score': np.max(compound_scores),
            'positive_count': distribution.get('positive_count', 0),
            'negative_count': distribution.get('negative_count', 0),
            'neutral_count': distribution.get('neutral_count', 0),
            'positive_percentage': distribution.get('positive_percent', 0),
            'negative_percentage': distribution.get('negative_percent', 0),
            'neutral_percentage': distribution.get('neutral_percent', 0),
            'dominant_sentiment': max(['positive', 'negative', 'neutral'],
                                     key=lambda x: distribution.get(f'{x}_count', 0))
        }

        return summary

    def plot_sentiment_comparison(self, comparisons: Dict,
                                 title: str = "Sentiment Comparison") -> plt.Figure:
        """
        Plot sentiment comparison between groups.

        Args:
            comparisons: Comparison dictionary from compare_sentiment_between_groups
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        groups = [comparisons['group1'], comparisons['group2']]
        group_names = [g['name'] for g in groups]

        # Compound score comparison
        compound_means = [g['mean_compound'] for g in groups]
        compound_stds = [g['std_compound'] for g in groups]

        ax1.bar(group_names, compound_means, yerr=compound_stds, capsize=5, color=['skyblue', 'lightcoral'])
        ax1.set_title('Average Compound Score', fontweight='bold')
        ax1.set_ylabel('Compound Score')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Sentiment distribution comparison
        sentiments = ['Positive', 'Negative', 'Neutral']
        group1_percentages = [groups[0]['positive_percent'], groups[0]['negative_percent'], groups[0]['neutral_percent']]
        group2_percentages = [groups[1]['positive_percent'], groups[1]['negative_percent'], groups[1]['neutral_percent']]

        x = np.arange(len(sentiments))
        width = 0.35

        ax2.bar(x - width/2, group1_percentages, width, label=group_names[0], color='lightgreen')
        ax2.bar(x + width/2, group2_percentages, width, label=group_names[1], color='lightcoral')
        ax2.set_title('Sentiment Distribution Comparison', fontweight='bold')
        ax2.set_ylabel('Percentage')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sentiments)
        ax2.legend()

        # Text count comparison
        text_counts = [g['count'] for g in groups]
        ax3.bar(group_names, text_counts, color=['skyblue', 'lightcoral'])
        ax3.set_title('Number of Texts Analyzed', fontweight='bold')
        ax3.set_ylabel('Count')

        # Statistical significance
        if comparisons['statistical_test']['significant_difference']:
            sig_text = f"Significant difference (p={comparisons['statistical_test']['p_value']:.4f})"
            color = 'green'
        else:
            sig_text = f"No significant difference (p={comparisons['statistical_test']['p_value']:.4f})"
            color = 'red'

        ax4.text(0.5, 0.5, sig_text, ha='center', va='center', fontsize=14,
                color=color, transform=ax4.transAxes, fontweight='bold')
        ax4.set_title('Statistical Test Result', fontweight='bold')
        ax4.axis('off')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig