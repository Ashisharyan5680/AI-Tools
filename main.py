import pytesseract
from PIL import Image
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline

# Step 1: Set Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ASUS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Step 2: Load Image
IMAGE_PATH = "image.png"

# Step 3: OCR Extraction
image = Image.open(IMAGE_PATH)
text = pytesseract.image_to_string(image)

print("\n RAW EXTRACTED TEXT:\n")
print(text)

# Step 4: Text Cleaning
clean_text = re.sub(r'\n+', ' ', text)
clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', clean_text)

print("\n CLEANED TEXT:\n")
print(clean_text)

# Step 5: Sentiment Analysis
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

sentiment = classifier(clean_text[:512])[0]

print("\n SENTIMENT:\n")
print(sentiment)

# Step 6: Keyword Extraction
vectorizer = CountVectorizer(stop_words='english', max_features=5)
X = vectorizer.fit_transform([clean_text])
keywords = vectorizer.get_feature_names_out()

print("\n KEYWORDS:\n")
print(keywords)

# Step 7: Summarization
summarizer = pipeline(
    "text-generation",
    model="google/flan-t5-small"
)

prompt = "Summarize this text: " + clean_text[:1000]

summary = summarizer(prompt, max_new_tokens=100)

print("\n SUMMARY:\n")
print(summary[0]['generated_text'])

# Step 8: Save Results to CSV
data = {
    "extracted_text": clean_text,
    "sentiment": sentiment['label'],
    "keywords": ", ".join(keywords),
    "summary": summary[0]['generated_text']
}

df = pd.DataFrame([data])
df.to_csv("results.csv", index=False)

print("\n Results saved to results.csv")
