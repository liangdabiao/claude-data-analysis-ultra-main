"""
Topic Analysis and Modeling

This module provides comprehensive topic analysis capabilities including
keyword extraction, topic modeling, and semantic analysis for content insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import warnings

class TopicAnalyzer:
    """
    Comprehensive topic analysis toolkit.

    Provides keyword extraction, topic modeling, and semantic analysis
    for content understanding and insight generation.
    """

    def __init__(self, language='english'):
        """
        Initialize the TopicAnalyzer.

        Args:
            language: Language for text processing (default: 'english')
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()

        # Load stopwords for specified language
        try:
            self.stop_words = set(stopwords.words(language))
        except OSError:
            # Fallback to English if language not found
            self.stop_words = set(stopwords.words('english'))

        # Add custom stopwords
        custom_stopwords = {'http', 'https', 'www', 'com', 'org', 'net', 'also', 'said', 'say'}
        self.stop_words.update(custom_stopwords)

    def extract_keywords_tfidf(self, texts: List[str],
                             top_k: int = 10,
                             max_features: int = 1000,
                             ngram_range: Tuple[int, int] = (1, 2)) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF scoring.

        Args:
            texts: List of texts to analyze
            top_k: Number of top keywords to return
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to consider

        Returns:
            List of (keyword, score) tuples
        """
        if not texts:
            return []

        # Combine texts for corpus-level analysis
        corpus = [' '.join(texts)]

        try:
            # TF-IDF vectorization
            tfidf = TfidfVectorizer(
                max_features=max_features,
                stop_words=list(self.stop_words),
                ngram_range=ngram_range,
                lowercase=True
            )
            tfidf_matrix = tfidf.fit_transform(corpus)

            # Get feature names and scores
            feature_names = tfidf.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]

            # Create keyword-score pairs
            keywords = [(feature_names[i], tfidf_scores[i])
                       for i in range(len(feature_names))
                       if tfidf_scores[i] > 0]

            # Sort by score and return top k
            keywords.sort(key=lambda x: x[1], reverse=True)
            return keywords[:top_k]

        except Exception as e:
            print(f"Error in TF-IDF keyword extraction: {e}")
            return []

    def extract_keywords_frequency(self, texts: List[str],
                                 top_k: int = 10,
                                 min_freq: int = 2) -> List[Tuple[str, int]]:
        """
        Extract keywords using frequency analysis.

        Args:
            texts: List of texts to analyze
            top_k: Number of top keywords to return
            min_freq: Minimum frequency threshold

        Returns:
            List of (keyword, frequency) tuples
        """
        if not texts:
            return []

        # Tokenize and process all texts
        all_words = []
        for text in texts:
            # Clean and tokenize
            from scripts.text_analyzer import TextAnalyzer
            analyzer = TextAnalyzer(language=self.language)
            tokens = analyzer.tokenize_text(text)
            all_words.extend(tokens)

        # Count word frequencies
        word_freq = Counter(all_words)

        # Filter by minimum frequency
        filtered_words = {word: freq for word, freq in word_freq.items()
                         if freq >= min_freq}

        # Return top k words
        return sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def perform_lda_topic_modeling(self, texts: List[str],
                                  num_topics: int = 5,
                                  max_features: int = 1000,
                                  passes: int = 10) -> Dict:
        """
        Perform topic modeling using Latent Dirichlet Allocation.

        Args:
            texts: List of texts to model
            num_topics: Number of topics to discover
            max_features: Maximum number of features
            passes: Number of passes through the data

        Returns:
            Dictionary with topic modeling results
        """
        if not texts or len(texts) < num_topics:
            return {}

        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                from scripts.text_analyzer import TextAnalyzer
                analyzer = TextAnalyzer(language=self.language)
                tokens = analyzer.tokenize_text(text)
                processed_texts.append(' '.join(tokens))

            # Remove empty texts
            processed_texts = [text for text in processed_texts if text.strip()]

            if len(processed_texts) < num_topics:
                return {}

            # Create document-term matrix
            count_vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2)
            )
            doc_term_matrix = count_vectorizer.fit_transform(processed_texts)

            # Fit LDA model
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                passes=passes
            )
            lda.fit(doc_term_matrix)

            # Get feature names
            feature_names = count_vectorizer.get_feature_names_out()

            # Extract topics
            topics = {}
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics[f'topic_{topic_idx}'] = {
                    'words': top_words,
                    'weights': [float(topic[i]) for i in top_words_idx]
                }

            # Transform documents to topic space
            doc_topic_distribution = lda.transform(doc_term_matrix)

            return {
                'topics': topics,
                'num_topics': num_topics,
                'feature_names': feature_names.tolist(),
                'doc_topic_distribution': doc_topic_distribution.tolist(),
                'perplexity': float(lda.perplexity(doc_term_matrix)),
                'log_likelihood': float(lda.score(doc_term_matrix))
            }

        except Exception as e:
            print(f"Error in LDA topic modeling: {e}")
            return {}

    def perform_nmf_topic_modeling(self, texts: List[str],
                                  num_topics: int = 5,
                                  max_features: int = 1000) -> Dict:
        """
        Perform topic modeling using Non-negative Matrix Factorization.

        Args:
            texts: List of texts to model
            num_topics: Number of topics to discover
            max_features: Maximum number of features

        Returns:
            Dictionary with topic modeling results
        """
        if not texts or len(texts) < num_topics:
            return {}

        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                from scripts.text_analyzer import TextAnalyzer
                analyzer = TextAnalyzer(language=self.language)
                tokens = analyzer.tokenize_text(text)
                processed_texts.append(' '.join(tokens))

            # Remove empty texts
            processed_texts = [text for text in processed_texts if text.strip()]

            if len(processed_texts) < num_topics:
                return {}

            # TF-IDF vectorization
            tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                lowercase=True
            )
            tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)

            # Fit NMF model
            nmf = NMF(n_components=num_topics, random_state=42)
            nmf.fit(tfidf_matrix)

            # Get feature names
            feature_names = tfidf_vectorizer.get_feature_names_out()

            # Extract topics
            topics = {}
            for topic_idx, topic in enumerate(nmf.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics[f'topic_{topic_idx}'] = {
                    'words': top_words,
                    'weights': [float(topic[i]) for i in top_words_idx]
                }

            # Transform documents to topic space
            doc_topic_distribution = nmf.transform(tfidf_matrix)

            return {
                'topics': topics,
                'num_topics': num_topics,
                'feature_names': feature_names.tolist(),
                'doc_topic_distribution': doc_topic_distribution.tolist(),
                'reconstruction_error': float(nmf.reconstruction_err_)
            }

        except Exception as e:
            print(f"Error in NMF topic modeling: {e}")
            return {}

    def cluster_texts(self, texts: List[str],
                     num_clusters: int = 5,
                     method: str = 'kmeans') -> Dict:
        """
        Cluster texts by similarity.

        Args:
            texts: List of texts to cluster
            num_clusters: Number of clusters to create
            method: Clustering method ('kmeans')

        Returns:
            Dictionary with clustering results
        """
        if not texts or len(texts) < num_clusters:
            return {}

        try:
            # Preprocess texts
            processed_texts = []
            for text in texts:
                from scripts.text_analyzer import TextAnalyzer
                analyzer = TextAnalyzer(language=self.language)
                cleaned_text = analyzer.clean_text(text)
                if cleaned_text.strip():
                    processed_texts.append(cleaned_text)

            if len(processed_texts) < num_clusters:
                return {}

            # TF-IDF vectorization
            tfidf = TfidfVectorizer(
                max_features=1000,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                lowercase=True
            )
            tfidf_matrix = tfidf.fit_transform(processed_texts)

            if method == 'kmeans':
                # K-means clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(tfidf_matrix)

                # Get cluster centers and feature names
                feature_names = tfidf.get_feature_names_out()
                cluster_centers = kmeans.cluster_centers_

                # Extract top terms for each cluster
                cluster_terms = {}
                for i in range(num_clusters):
                    top_indices = cluster_centers[i].argsort()[-10:][::-1]
                    cluster_terms[i] = [feature_names[idx] for idx in top_indices]

                return {
                    'cluster_labels': cluster_labels.tolist(),
                    'cluster_terms': cluster_terms,
                    'num_clusters': num_clusters,
                    'feature_names': feature_names.tolist(),
                    'inertia': float(kmeans.inertia_)
                }

        except Exception as e:
            print(f"Error in text clustering: {e}")
            return {}

    def analyze_topic_trends(self, texts: List[str],
                           timestamps: List[str] = None,
                           time_window: str = 'daily') -> pd.DataFrame:
        """
        Analyze topic trends over time.

        Args:
            texts: List of texts to analyze
            timestamps: List of timestamps corresponding to texts
            time_window: Time aggregation window ('daily', 'weekly', 'monthly')

        Returns:
            DataFrame with topic trends
        """
        if not texts:
            return pd.DataFrame()

        # Perform topic modeling
        topic_results = self.perform_lda_topic_modeling(texts, num_topics=5)

        if not topic_results:
            return pd.DataFrame()

        doc_topic_dist = topic_results['doc_topic_distribution']

        # Create DataFrame with topic distributions
        df = pd.DataFrame(doc_topic_dist)
        df.columns = [f'topic_{i}' for i in range(topic_results['num_topics'])]

        # Add timestamps if provided
        if timestamps:
            df['timestamp'] = pd.to_datetime(timestamps)
            df.set_index('timestamp', inplace=True)

            # Aggregate by time window
            if time_window == 'daily':
                df_trends = df.resample('D')
            elif time_window == 'weekly':
                df_trends = df.resample('W')
            elif time_window == 'monthly':
                df_trends = df.resample('M')
            else:
                df_trends = df.resample('D')

            # Calculate topic trends
            trends = df_trends.mean()

            return trends.reset_index()

        else:
            # Simple index-based trends
            return df.reset_index(drop=True)

    def find_similar_content(self, query_text: str,
                           text_corpus: List[str],
                           top_k: int = 5,
                           method: str = 'tfidf') -> List[Tuple[int, float]]:
        """
        Find most similar content to a query text.

        Args:
            query_text: Query text
            text_corpus: List of texts to search through
            top_k: Number of most similar texts to return
            method: Similarity method ('tfidf')

        Returns:
            List of (index, similarity_score) tuples
        """
        if not query_text or not text_corpus:
            return []

        try:
            # Prepare texts
            all_texts = [query_text] + text_corpus
            processed_texts = []

            for text in all_texts:
                from scripts.text_analyzer import TextAnalyzer
                analyzer = TextAnalyzer(language=self.language)
                cleaned_text = analyzer.clean_text(text)
                if cleaned_text.strip():
                    processed_texts.append(cleaned_text)

            if len(processed_texts) < 2:
                return []

            if method == 'tfidf':
                # TF-IDF vectorization
                tfidf = TfidfVectorizer(
                    max_features=1000,
                    stop_words=list(self.stop_words),
                    ngram_range=(1, 2),
                    lowercase=True
                )
                tfidf_matrix = tfidf.fit_transform(processed_texts)

                # Calculate cosine similarity
                similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

                # Get top k most similar texts
                top_indices = similarities.argsort()[-top_k:][::-1]

                return [(int(idx), float(similarities[idx])) for idx in top_indices]

        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            return []

    def create_topic_wordcloud(self, texts: List[str],
                             topic_id: int = 0,
                             method: str = 'lda') -> WordCloud:
        """
        Create a word cloud for a specific topic.

        Args:
            texts: List of texts to analyze
            topic_id: Topic ID to visualize
            method: Topic modeling method ('lda' or 'nmf')

        Returns:
            WordCloud object
        """
        try:
            if method == 'lda':
                topic_results = self.perform_lda_topic_modeling(texts, num_topics=topic_id + 1)
            elif method == 'nmf':
                topic_results = self.perform_nmf_topic_modeling(texts, num_topics=topic_id + 1)
            else:
                raise ValueError("Method must be 'lda' or 'nmf'")

            if not topic_results:
                return None

            # Get topic words and weights
            topic_key = f'topic_{topic_id}'
            if topic_key not in topic_results['topics']:
                return None

            topic_data = topic_results['topics'][topic_key]
            words = topic_data['words']
            weights = topic_data['weights']

            # Create word frequency dictionary
            word_freq = {word: weight for word, weight in zip(words, weights)}

            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate_from_frequencies(word_freq)

            return wordcloud

        except Exception as e:
            print(f"Error creating topic word cloud: {e}")
            return None

    def plot_topic_distribution(self, topic_results: Dict,
                               title: str = "Topic Distribution") -> plt.Figure:
        """
        Plot topic distribution across documents.

        Args:
            topic_results: Results from topic modeling
            title: Plot title

        Returns:
            Matplotlib figure object
        """
        if not topic_results or 'doc_topic_distribution' not in topic_results:
            return None

        try:
            import matplotlib.pyplot as plt

            # Create DataFrame from topic distribution
            doc_topic_dist = pd.DataFrame(topic_results['doc_topic_distribution'])
            doc_topic_dist.columns = [f'Topic {i}' for i in range(topic_results['num_topics'])]

            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            # Plot 1: Topic prevalence (average across all documents)
            avg_topic_dist = doc_topic_dist.mean()
            axes[0].bar(avg_topic_dist.index, avg_topic_dist.values, color='skyblue')
            axes[0].set_title('Average Topic Prevalence', fontweight='bold')
            axes[0].set_ylabel('Average Probability')
            axes[0].tick_params(axis='x', rotation=45)

            # Plot 2: Topic distribution histogram
            dominant_topics = doc_topic_dist.idxmax(axis=1)
            topic_counts = dominant_topics.value_counts()
            axes[1].bar(topic_counts.index, topic_counts.values, color='lightgreen')
            axes[1].set_title('Dominant Topic Distribution', fontweight='bold')
            axes[1].set_ylabel('Number of Documents')
            axes[1].tick_params(axis='x', rotation=45)

            # Plot 3: Topic correlation heatmap
            topic_corr = doc_topic_dist.corr()
            im = axes[2].imshow(topic_corr.values, cmap='coolwarm', aspect='auto')
            axes[2].set_xticks(range(len(topic_corr.columns)))
            axes[2].set_yticks(range(len(topic_corr.columns)))
            axes[2].set_xticklabels(topic_corr.columns, rotation=45)
            axes[2].set_yticklabels(topic_corr.columns)
            axes[2].set_title('Topic Correlation Matrix', fontweight='bold')

            # Add colorbar
            plt.colorbar(im, ax=axes[2])

            # Plot 4: Top words for each topic
            if 'topics' in topic_results:
                topic_words = []
                for i in range(topic_results['num_topics']):
                    topic_key = f'topic_{i}'
                    if topic_key in topic_results['topics']:
                        top_words = ', '.join(topic_results['topics'][topic_key]['words'][:5])
                        topic_words.append(f"Topic {i}: {top_words}")

                y_pos = range(len(topic_words))
                axes[3].barh(y_pos, [1] * len(topic_words), color='orange')
                axes[3].set_yticks(y_pos)
                axes[3].set_yticklabels(topic_words)
                axes[3].set_title('Top Words by Topic', fontweight='bold')
                axes[3].set_xlabel('Relative Importance')
                axes[3].tick_params(axis='y')

            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()

            return fig

        except Exception as e:
            print(f"Error plotting topic distribution: {e}")
            return None

    def get_topic_summary(self, topic_results: Dict) -> Dict:
        """
        Get a comprehensive summary of topic modeling results.

        Args:
            topic_results: Results from topic modeling

        Returns:
            Dictionary with topic summary
        """
        if not topic_results:
            return {}

        summary = {
            'num_topics': topic_results.get('num_topics', 0),
            'total_documents': len(topic_results.get('doc_topic_distribution', [])),
            'dominant_topics': [],
            'topic_coherence': 0.0,
            'model_quality': {}
        }

        # Calculate dominant topics
        doc_topic_dist = topic_results.get('doc_topic_distribution', [])
        if doc_topic_dist:
            dominant_topics = [np.argmax(doc) for doc in doc_topic_dist]
            topic_counts = Counter(dominant_topics)
            summary['dominant_topics'] = [
                {'topic_id': topic_id, 'document_count': count, 'percentage': count/len(doc_topic_dist)*100}
                for topic_id, count in topic_counts.items()
            ]

        # Add model quality metrics
        if 'perplexity' in topic_results:
            summary['model_quality']['perplexity'] = topic_results['perplexity']
        if 'log_likelihood' in topic_results:
            summary['model_quality']['log_likelihood'] = topic_results['log_likelihood']
        if 'reconstruction_error' in topic_results:
            summary['model_quality']['reconstruction_error'] = topic_results['reconstruction_error']

        return summary