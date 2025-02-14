print("Loading libraries")
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
print("Loading models")

emotion_model_name = "SamLowe/roberta-base-go_emotions"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)
print("Success loading!")

nlp = spacy.load("en_core_web_sm")


def classify_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = emotion_model(**inputs)
    probs = softmax(outputs.logits, dim=-1)[0]

    id2label = emotion_model.config.id2label
    emotions = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    target_emotions = {
        'sadness': 'sad',
        'joy': 'joyful',
        'anger': 'anger',
        'fear': 'fear',
        'love': 'love',
        'surprise': 'surprise'
    }

    max_emotion = max((k for k in emotions.keys() if k in target_emotions),
                      key=lambda k: emotions[k])

    return target_emotions[max_emotion], emotions


def analyze_anger_cause(text):
    doc = nlp(text)
    verbs = []
    nouns = []

    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            verbs.append(token.lemma_)
        elif token.pos_ == "NOUN":
            nouns.append(token.text)

    if verbs and nouns:
        return f"{verbs[0]} {nouns[0]}"
    return "unknown reason"


def analyze_user_input():
    print("\n" + "-" * 50 + "\n")
    while True:
        user_input = input("Enter the message in English (or '0' for exit): ")
        if user_input == '0':
            print("Good Luck!")
            break

        emotion, scores = classify_emotion(user_input)
        print(f"\nEmotion: {emotion}")

        reason = analyze_anger_cause(user_input)
        if emotion == 'anger':
            print(f"The reason for the anger: {reason}")

        responses = {
            "sad": "Are you sad? Don't you want to listen to Vova Solodkov - Barbulka!",
            "love": "Are you in love? I congratulate you!",
            "surprise": "Are you surprised? I congratulate you!",
            "joyful": "Are you feeling joy? Go for a walk!",
            "fear": "Are you fear? I congratulate you!",
            "anger": f"Try not to think about {reason}, think about all the good things!"
        }

        print(f"Accuracy: {max(scores.values()) * 100:.2f}")
        print("\n", responses.get(emotion))
        print("\n" + "-" * 50 + "\n")


analyze_user_input()
