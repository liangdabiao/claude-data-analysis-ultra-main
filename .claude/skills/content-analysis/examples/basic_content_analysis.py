"""
Basic Content Analysis Example

This example demonstrates how to use the Content Analysis skill
to perform traditional NLP analysis on text data.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from scripts.text_analyzer import TextAnalyzer, SocialMediaAnalyzer
from scripts.sentiment_analyzer import SentimentAnalyzer
from scripts.topic_analyzer import TopicAnalyzer
from scripts.content_visualizer import ContentVisualizer

def create_sample_data():
    """
    Create sample text data for analysis demonstration.
    """
    sample_texts = [
        "I absolutely love this new product! It's amazing and works perfectly.",
        "The customer service was terrible. I'm very disappointed with the experience.",
        "This is an okay product. Nothing special but it does the job.",
        "Outstanding quality and great value for money. Highly recommended!",
        "Poor packaging and the item arrived damaged. Not happy at all.",
        "Average quality, meets expectations but doesn't exceed them.",
        "Fantastic experience from start to finish. Will definitely buy again!",
        "The product broke after just one week. Complete waste of money.",
        "Good product for the price. Some minor issues but overall satisfied.",
        "Excellent build quality and attention to detail. Worth every penny!",
        "Slow shipping and the product doesn't match the description.",
        "Decent product, but could be improved with better features.",
        "Absolutely love it! Best purchase I've made this year.",
        "Mediocre quality, expected more for this price point.",
        "Great customer support and fast resolution of my issues.",
        "Product stopped working after a few days. Very frustrated.",
        "Solid product with good performance. Meets my needs well.",
        "Terrible user manual and confusing setup process.",
        "Impressed with the quality and fast delivery. Excellent service!",
        "Not worth the price. There are better alternatives available."
    ]

    # Create DataFrame
    df = pd.DataFrame({
        'text': sample_texts,
        'category': np.random.choice(['Product', 'Service', 'Shipping'], len(sample_texts)),
        'source': np.random.choice(['Twitter', 'Review', 'Email'], len(sample_texts)),
        'date': pd.date_range('2024-01-01', periods=len(sample_texts), freq='D')
    })

    return df

def demonstrate_text_analysis():
    """
    Demonstrate basic text analysis capabilities.
    """
    print("=== Basic Text Analysis Demo ===\n")

    # Sample text
    sample_text = "I absolutely love this new product! It's amazing and works perfectly. The customer service was excellent too."

    # Initialize analyzer
    analyzer = TextAnalyzer()

    # Clean text
    cleaned = analyzer.clean_text(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Cleaned text: {cleaned}")

    # Tokenize
    tokens = analyzer.tokenize_text(sample_text)
    print(f"\nTokens: {tokens[:10]}...")  # Show first 10 tokens

    # Extract keywords
    keywords = analyzer.extract_keywords(sample_text, top_k=5)
    print(f"\nTop keywords: {keywords}")

    # Calculate readability
    readability = analyzer.calculate_readability(sample_text)
    print(f"\nReadability metrics: {readability}")

    # Get text statistics
    stats = analyzer.get_text_statistics(sample_text)
    print(f"\nText statistics: {stats}")

def demonstrate_sentiment_analysis():
    """
    Demonstrate sentiment analysis capabilities.
    """
    print("\n=== Sentiment Analysis Demo ===\n")

    # Sample texts with different sentiments
    texts = [
        "I love this product! It's absolutely amazing.",
        "This is terrible. I hate it and want my money back.",
        "The product is okay. Nothing special but does the job.",
        "Outstanding quality and excellent customer service!",
        "Poor quality and terrible customer support."
    ]

    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()

    # Analyze each text
    results = []
    for text in texts:
        result = analyzer.analyze_sentiment_vader(text)
        result['text'] = text
        results.append(result)

        print(f"Text: {text}")
        print(f"Sentiment: {result['label']} (Compound: {result['compound']:.3f})")
        print(f"Positive: {result['positive']:.3f}, Negative: {result['negative']:.3f}, Neutral: {result['neutral']:.3f}")
        print()

    # Get sentiment distribution
    distribution = analyzer.classify_sentiment_distribution(texts)
    print("Sentiment Distribution:")
    print(f"Positive: {distribution.get('positive_percent', 0):.1f}% ({distribution.get('positive_count', 0)} texts)")
    print(f"Negative: {distribution.get('negative_percent', 0):.1f}% ({distribution.get('negative_count', 0)} texts)")
    print(f"Neutral: {distribution.get('neutral_percent', 0):.1f}% ({distribution.get('neutral_count', 0)} texts)")

def demonstrate_topic_analysis():
    """
    Demonstrate topic analysis capabilities.
    """
    print("\n=== Topic Analysis Demo ===\n")

    # Sample texts about different topics
    texts = [
        "The new smartphone has amazing camera quality and fast performance.",
        "Customer service was excellent and resolved my issue quickly.",
        "Fast shipping and great packaging. Product arrived in perfect condition.",
        "The laptop battery life is impressive and the screen is very clear.",
        "Technical support was very helpful and knowledgeable.",
        "The delivery was delayed but the product quality is good.",
        "Great value for money and the features are exactly what I needed.",
        "Had to return the item due to manufacturing defects.",
        "The tablet is lightweight and perfect for travel.",
        "Prices are competitive but the warranty coverage is limited."
    ]

    # Initialize topic analyzer
    analyzer = TopicAnalyzer()

    # Extract keywords using TF-IDF
    tfidf_keywords = analyzer.extract_keywords_tfidf(texts, top_k=10)
    print("TF-IDF Keywords:")
    for keyword, score in tfidf_keywords:
        print(f"  {keyword}: {score:.4f}")

    # Extract keywords using frequency
    freq_keywords = analyzer.extract_keywords_frequency(texts, top_k=10)
    print(f"\nFrequency Keywords:")
    for keyword, freq in freq_keywords:
        print(f"  {keyword}: {freq}")

    # Perform topic modeling
    print(f"\nPerforming topic modeling...")
    topic_results = analyzer.perform_lda_topic_modeling(texts, num_topics=3)

    if topic_results and 'topics' in topic_results:
        print(f"Discovered {topic_results['num_topics']} topics:")
        for topic_key, topic_data in topic_results['topics'].items():
            print(f"\n{topic_key.upper()}:")
            print(f"  Top words: {', '.join(topic_data['words'][:8])}")

def demonstrate_social_media_analysis():
    """
    Demonstrate social media specific analysis.
    """
    print("\n=== Social Media Analysis Demo ===\n")

    # Sample social media posts
    social_posts = [
        "Just tried the new #iPhone15 and it's amazing! 📱 #Apple #Tech",
        "Terrible customer service @CompanyXYZ. Waited 2 hours for support! 😤",
        "Check out our latest sale! 50% off everything. #Shopping #Deals",
        "@JohnDoe thanks for the recommendation! The product works great 👍",
        "Retweet if you agree that we need better environmental policies! 🌍 #Climate",
        "New blog post: 10 tips for better productivity. Link in bio! ✨",
        "Can't believe they cancelled my order without notification! So frustrated 😠",
        "Happy holidays everyone! 🎄 Wishing you joy and peace this season."
    ]

    # Initialize social media analyzer
    analyzer = SocialMediaAnalyzer()

    # Analyze each post
    print("Social Media Post Analysis:")
    for i, post in enumerate(social_posts, 1):
        hashtags = analyzer.extract_hashtags(post)
        mentions = analyzer.extract_mentions(post)
        engagement = analyzer.count_engagement_indicators(post)
        platform = analyzer.detect_social_media_platform(post)

        print(f"\nPost {i}:")
        print(f"  Text: {post}")
        print(f"  Platform: {platform}")
        print(f"  Hashtags: {hashtags}")
        print(f"  Mentions: {mentions}")
        print(f"  Engagement indicators: {engagement}")

def demonstrate_content_clustering():
    """
    Demonstrate content clustering capabilities.
    """
    print("\n=== Content Clustering Demo ===\n")

    # Sample texts for clustering
    texts = [
        "The smartphone has excellent camera and battery life.",
        "Great customer service and fast response time.",
        "The tablet offers good value for money.",
        "Technical support team was very helpful.",
        "Laptop performance is outstanding for gaming.",
        "Shipping was fast and packaging was secure.",
        "The phone's user interface is intuitive.",
        "Had issues with product registration.",
        "Great warranty coverage and after-sales service.",
        "The device is lightweight and portable."
    ]

    # Initialize analyzer
    analyzer = TopicAnalyzer()

    # Perform clustering
    print("Performing text clustering...")
    cluster_results = analyzer.cluster_texts(texts, num_clusters=3)

    if cluster_results and 'cluster_terms' in cluster_results:
        print(f"Cluster Analysis Results:")
        print(f"Number of clusters: {cluster_results['num_clusters']}")
        print(f"Inertia (within-cluster sum of squares): {cluster_results['inertia']:.2f}")

        print(f"\nTop terms by cluster:")
        for cluster_id, terms in cluster_results['cluster_terms'].items():
            print(f"  Cluster {cluster_id}: {', '.join(terms[:5])}")

def create_visualizations():
    """
    Create sample visualizations.
    """
    print("\n=== Creating Visualizations ===\n")

    # Sample sentiment data
    sentiment_data = [
        {'label': 'Positive', 'compound': 0.8},
        {'label': 'Negative', 'compound': -0.6},
        {'label': 'Neutral', 'compound': 0.1},
        {'label': 'Positive', 'compound': 0.9},
        {'label': 'Positive', 'compound': 0.7},
        {'label': 'Negative', 'compound': -0.8},
        {'label': 'Neutral', 'compound': 0.0},
        {'label': 'Positive', 'compound': 0.6}
    ]

    # Sample topic results
    topic_results = {
        'num_topics': 3,
        'topics': {
            'topic_0': {
                'words': ['product', 'quality', 'excellent', 'great', 'value'],
                'weights': [0.15, 0.12, 0.10, 0.09, 0.08]
            },
            'topic_1': {
                'words': ['customer', 'service', 'support', 'help', 'response'],
                'weights': [0.14, 0.13, 0.11, 0.09, 0.08]
            },
            'topic_2': {
                'words': ['shipping', 'delivery', 'fast', 'packaging', 'condition'],
                'weights': [0.16, 0.14, 0.11, 0.09, 0.07]
            }
        },
        'doc_topic_distribution': [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7]
        ]
    }

    # Initialize visualizer
    visualizer = ContentVisualizer()

    # Create sentiment distribution plot
    print("Creating sentiment distribution visualization...")
    sentiment_fig = visualizer.plot_sentiment_distribution(sentiment_data)
    if sentiment_fig:
        sentiment_fig.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
        print("Saved: sentiment_distribution.png")

    # Create topic modeling visualization
    print("Creating topic modeling visualization...")
    topic_fig = visualizer.plot_topic_modeling_results(topic_results)
    if topic_fig:
        topic_fig.savefig('topic_modeling_results.png', dpi=300, bbox_inches='tight')
        print("Saved: topic_modeling_results.png")

    # Create word cloud
    print("Creating word cloud...")
    sample_text = """
    product quality excellent value customer service great support
    shipping delivery fast packaging condition warranty coverage
    technical helpful responsive resolution satisfaction recommendation
    """
    wordcloud_fig = visualizer.create_word_cloud(sample_text, title="Content Analysis Word Cloud")
    if wordcloud_fig:
        wordcloud_fig.savefig('content_wordcloud.png', dpi=300, bbox_inches='tight')
        print("Saved: content_wordcloud.png")

def main():
    """
    Run complete basic content analysis example.
    """
    print("Content Analysis Skill - Basic Analysis Example")
    print("=" * 50)

    # Create sample data
    print("\n1. Creating sample data...")
    df = create_sample_data()
    print(f"   Created dataset with {len(df)} text samples")

    # Demonstrate text analysis
    print("\n2. Demonstrating text analysis...")
    demonstrate_text_analysis()

    # Demonstrate sentiment analysis
    print("\n3. Demonstrating sentiment analysis...")
    demonstrate_sentiment_analysis()

    # Demonstrate topic analysis
    print("\n4. Demonstrating topic analysis...")
    demonstrate_topic_analysis()

    # Demonstrate social media analysis
    print("\n5. Demonstrating social media analysis...")
    demonstrate_social_media_analysis()

    # Demonstrate content clustering
    print("\n6. Demonstrating content clustering...")
    demonstrate_content_clustering()

    # Create visualizations
    print("\n7. Creating visualizations...")
    create_visualizations()

    print("\n" + "=" * 50)
    print("Basic Content Analysis Example Complete!")
    print("\nGenerated files:")
    print("  - sentiment_distribution.png")
    print("  - topic_modeling_results.png")
    print("  - content_wordcloud.png")

if __name__ == "__main__":
    main()