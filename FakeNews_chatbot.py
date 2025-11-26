import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import random

# --- CONFIGURATION ---
nltk.download('punkt')

# ==========================================
# TASK 1: CHATBOT INTENT DETECTION
# ==========================================
class ChatbotAgent:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.classifier = LogisticRegression()
        self.confidence_threshold = 0.55  # If prob is lower, trigger fallback
        
        # Pre-defined responses
        self.responses = {
            'greeting': "Hello! How can I assist you today?",
            'query': "We are open Mon-Fri from 9 AM to 5 PM. Our office is in Tech Park.",
            'feedback': "Thank you for your feedback! We will use it to improve.",
            'fallback': "I'm sorry, I didn't quite catch that. Could you rephrase?"
        }

    def train(self):
        """Generates synthetic data and trains the intent classifier."""
        print("--- Training Chatbot ---")
        # Synthetic Training Data
        data = {
            'text': [
                "Hi there", "Hello", "Good morning", "Hey", "Greetings",
                "What are your hours?", "When do you open?", "Location?", "Where is the office?", "Help me find you",
                "Great service", "I hated the experience", "Terrible food", "Awesome job", "Not happy",
                "Hi", "Good evening", "What time do you close?", "Bad service", "Lovely place"
            ],
            'intent': [
                'greeting', 'greeting', 'greeting', 'greeting', 'greeting',
                'query', 'query', 'query', 'query', 'query',
                'feedback', 'feedback', 'feedback', 'feedback', 'feedback',
                'greeting', 'greeting', 'query', 'feedback', 'feedback'
            ]
        }
        
        X = self.vectorizer.fit_transform(data['text'])
        self.classifier.fit(X, data['intent'])
        print("Chatbot trained successfully on 3 intents.")

    def get_response(self, user_input):
        """Predicts intent and returns a response. Handles low confidence."""
        # Vectorize input
        vec_input = self.vectorizer.transform([user_input])
        
        # Get probabilities
        probs = self.classifier.predict_proba(vec_input)[0]
        max_prob = np.max(probs)
        pred_index = np.argmax(probs)
        predicted_intent = self.classifier.classes_[pred_index]
        
        # Fallback Logic
        if max_prob < self.confidence_threshold:
            return f"{self.responses['fallback']} (Confidence: {max_prob:.2f})"
        
        return f"{self.responses[predicted_intent]} (Intent: {predicted_intent}, Conf: {max_prob:.2f})"

# ==========================================
# TASK 2: FAKE NEWS DETECTION
# ==========================================
class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()

    def load_and_train(self):
        """Generates synthetic news data and trains the model."""
        print("\n--- Training Fake News Detector ---")
        
        # Synthetic Data Generation
        # Feature Note: Fake news often uses caps, exclamation marks, and sensational words.
        real_headlines = [
            "Stock markets close higher today amid tech rally.",
            "President to visit Europe next week for summit.",
            "New bridge construction approved by city council.",
            "Scientists discover new species of deep sea fish.",
            "Local library extends opening hours for students.",
            "Weather forecast predicts rain for the weekend.",
            "Hospital capabilities expanded with new wing.",
            "Education minister announces new policy reforms.",
            "Traffic delays expected on the main highway.",
            "Company reports 5% increase in annual revenue."
        ] * 10
        
        fake_headlines = [
            "ALIENS LAND IN NEW YORK!! GOVERNMENT HIDING TRUTH!",
            "Drink this magical water to cure all diseases instantly!",
            "You won't believe what this celebrity did! SHOCKING!",
            "Scientists confirm the earth is actually flat.",
            "Millionaire gives away all money to random stranger!",
            "Shark found swimming in local swimming pool!!",
            "Flying cars are finally here and cost $100!",
            "Eating dirt is the new secret to eternal youth.",
            "Zombies spotted in downtown metro station!",
            "SECRET PLOT REVEALED: The moon is a hologram!"
        ] * 10

        df = pd.DataFrame({
            'text': real_headlines + fake_headlines,
            'label': ['Real'] * 100 + ['Fake'] * 100
        })

        # Preprocessing & Vectorization
        X = self.vectorizer.fit_transform(df['text'])
        y = df['label']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\nFake News Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.model

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- Execute Task 1: Chatbot ---
    bot = ChatbotAgent()
    bot.train()
    
    print("\n--- Chatbot Demo (Type 'exit' to stop) ---")
    test_queries = ["Hello there", "Where are you located?", "Your service was terrible", "Do you sell pizza?"]
    
    for query in test_queries:
        print(f"User: {query}")
        print(f"Bot:  {bot.get_response(query)}")
        
    # --- Execute Task 2: Fake News ---
    detector = FakeNewsDetector()
    detector.load_and_train()