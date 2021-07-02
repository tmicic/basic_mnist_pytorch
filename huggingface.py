from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk import sent_tokenize
 

# tokenizer = 

# model = AutoModelForSequenceClassification.from_pretrained("typeform/distilbert-base-uncased-mnli")

# text = "Replace me by any text you'd like."
# candidate_labels = ["normal", "not normal"]
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)

# exit()

from transformers import ZeroShotClassificationPipeline as pipeline
classifier = pipeline(model=AutoModelForSequenceClassification.from_pretrained("typeform/distilbert-base-uncased-mnli"),
                        tokenizer=AutoTokenizer.from_pretrained("typeform/distilbert-base-uncased-mnli"),
                        framework='pt')
                        
# sentence = "My friend holds a Msc. in Computer Science."
# print sent_tokenize(sentence)
# results = classifier(
#   "The mass within the left upper lobe has increased in size.", 
#   ["normal", "neutral", "not normal"]
# )

# print(results)

print(classifier.tokenizer.vocab.keys())