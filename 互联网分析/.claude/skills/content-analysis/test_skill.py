#!/usr/bin/env python3
"""
Test script for the Content Analysis Skill

This script tests the basic functionality of all modules in the content analysis skill.
"""

import sys
import os
import traceback

def test_imports():
    """Test importing all modules."""
    print("Testing imports...")

    try:
        from scripts.text_analyzer import TextAnalyzer, SocialMediaAnalyzer
        print("✓ TextAnalyzer imported successfully")
        print("✓ SocialMediaAnalyzer imported successfully")
    except Exception as e:
        print(f"✗ TextAnalyzer import failed: {e}")
        return False

    try:
        from scripts.sentiment_analyzer import SentimentAnalyzer
        print("✓ SentimentAnalyzer imported successfully")
    except Exception as e:
        print(f"✗ SentimentAnalyzer import failed: {e}")
        return False

    try:
        from scripts.topic_analyzer import TopicAnalyzer
        print("✓ TopicAnalyzer imported successfully")
    except Exception as e:
        print(f"✗ TopicAnalyzer import failed: {e}")
        return False

    try:
        from scripts.content_visualizer import ContentVisualizer
        print("✓ ContentVisualizer imported successfully")
    except Exception as e:
        print(f"✗ ContentVisualizer import failed: {e}")
        return False

    # Optional LLM analyzer (may fail without API keys)
    try:
        from scripts.llm_analyzer import LLMAnalyzer
        print("✓ LLMAnalyzer imported successfully")
    except Exception as e:
        print(f"⚠ LLMAnalyzer import failed (expected without API keys): {e}")

    return True

def test_basic_functionality():
    """Test basic functionality of core modules."""
    print("\nTesting basic functionality...")

    try:
        # Test TextAnalyzer
        from scripts.text_analyzer import TextAnalyzer
        analyzer = TextAnalyzer()

        test_text = "Hello World! This is a test of the content analysis skill."
        cleaned = analyzer.clean_text(test_text)
        print(f"✓ Text cleaning: '{test_text}' -> '{cleaned}'")

        tokens = analyzer.tokenize_text(test_text)
        print(f"✓ Tokenization: {tokens}")

        keywords = analyzer.extract_keywords(test_text, top_k=5)
        print(f"✓ Keyword extraction: {keywords}")

    except Exception as e:
        print(f"✗ TextAnalyzer functionality test failed: {e}")
        return False

    try:
        # Test SentimentAnalyzer
        from scripts.sentiment_analyzer import SentimentAnalyzer
        sentiment_analyzer = SentimentAnalyzer()

        sentiment_result = sentiment_analyzer.analyze_sentiment_vader(test_text)
        print(f"✓ Sentiment analysis: {sentiment_result['label']} ({sentiment_result['compound']:.3f})")

    except Exception as e:
        print(f"✗ SentimentAnalyzer functionality test failed: {e}")
        return False

    try:
        # Test TopicAnalyzer
        from scripts.topic_analyzer import TopicAnalyzer
        topic_analyzer = TopicAnalyzer()

        texts = [
            "I love this new smartphone! The camera is amazing.",
            "The customer service was terrible and unhelpful.",
            "Great product quality and fast shipping. Very satisfied!",
            "Poor packaging and the item arrived damaged."
        ]

        keywords = topic_analyzer.extract_keywords_tfidf(texts, top_k=5)
        print(f"✓ TF-IDF keyword extraction: {[kw[0] for kw in keywords[:5]]}")

    except Exception as e:
        print(f"✗ TopicAnalyzer functionality test failed: {e}")
        return False

    try:
        # Test ContentVisualizer
        from scripts.content_visualizer import ContentVisualizer
        visualizer = ContentVisualizer()
        print("✓ ContentVisualizer initialization successful")

    except Exception as e:
        print(f"✗ ContentVisualizer functionality test failed: {e}")
        return False

    return True

def test_sample_data():
    """Test loading and processing sample data."""
    print("\nTesting sample data processing...")

    try:
        import pandas as pd
        import json

        # Test CSV data
        csv_path = "examples/sample_data/sample_reviews.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded CSV data with {len(df)} rows and columns: {list(df.columns)}")

            # Test analysis on sample data
            from scripts.sentiment_analyzer import SentimentAnalyzer
            analyzer = SentimentAnalyzer()

            sample_texts = df['text'].tolist()[:3]  # Test first 3 texts
            results = analyzer.analyze_batch_vader(sample_texts)
            print(f"✓ Analyzed {len(results)} sample texts")

        else:
            print(f"⚠ Sample CSV file not found: {csv_path}")

        # Test JSON data
        json_path = "examples/sample_data/sample_social_media.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                social_data = json.load(f)
            print(f"✓ Loaded JSON data with {len(social_data)} social media posts")

            # Test social media analysis
            from scripts.text_analyzer import SocialMediaAnalyzer
            sm_analyzer = SocialMediaAnalyzer()

            sample_post = social_data[0]['text']
            hashtags = sm_analyzer.extract_hashtags(sample_post)
            mentions = sm_analyzer.extract_mentions(sample_post)
            print(f"✓ Extracted hashtags: {hashtags}, mentions: {mentions}")

        else:
            print(f"⚠ Sample JSON file not found: {json_path}")

    except Exception as e:
        print(f"✗ Sample data processing test failed: {e}")
        return False

    return True

def test_integration():
    """Test integration between modules."""
    print("\nTesting module integration...")

    try:
        # Test complete workflow
        from scripts.text_analyzer import TextAnalyzer
        from scripts.sentiment_analyzer import SentimentAnalyzer
        from scripts.topic_analyzer import TopicAnalyzer

        # Sample data
        texts = [
            "I absolutely love this new product! It's amazing and works perfectly.",
            "Terrible customer service. I'm very disappointed with the experience.",
            "This is an okay product. Nothing special but it does the job.",
            "Outstanding quality and great value for money. Highly recommended!"
        ]

        # Text preprocessing
        text_analyzer = TextAnalyzer()
        cleaned_texts = [text_analyzer.clean_text(text) for text in texts]
        print(f"✓ Preprocessed {len(cleaned_texts)} texts")

        # Sentiment analysis
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_results = sentiment_analyzer.analyze_batch_vader(texts)
        print(f"✓ Performed sentiment analysis on {len(sentiment_results)} texts")

        # Topic analysis
        topic_analyzer = TopicAnalyzer()
        topic_results = topic_analyzer.extract_keywords_tfidf(texts, top_k=5)
        print(f"✓ Extracted top topics: {[kw[0] for kw in topic_results]}")

        # Get summary
        sentiment_summary = sentiment_analyzer.get_sentiment_summary(texts)
        print(f"✓ Generated sentiment summary: {sentiment_summary['dominant_sentiment']} dominant")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False

    return True

def main():
    """Run all tests."""
    print("Content Analysis Skill - Test Suite")
    print("=" * 50)

    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality Tests", test_basic_functionality),
        ("Sample Data Tests", test_sample_data),
        ("Integration Tests", test_integration)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))

        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Content Analysis Skill is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)