from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

texts = [
    "I love this movie!",
    "The plot was predictable.",
    "The acting ws brilliant.",
    "I didn't enjoy watching it.",
    "The film was a disappointment."
]
labels = ["positive", "negative", "positive", "negative", "negative"]

vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(texts)
sorted_vocab = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[0])

print(f"\nTF-IDF Vocabulary sorted:\n{sorted(vectorizer.vocabulary_.items(), key=lambda x: x[0])}")  # A dictionary of terms with their indices
print(f"\nFeature Names (terms in the vocabulary):\n{vectorizer.get_feature_names_out()}")  # An array of feature names
print(f"\nTerm Frequencies in Documents:\n{features.toarray()}")  # Dense matrix representation

classifier = LinearSVC()
classifier.fit(features, labels)

new_text = "The movie exceeded my expectations!"
new_features = vectorizer.transform([new_text])
predicted_label = classifier.predict(new_features)
print(f"Predicted label: {predicted_label}")
