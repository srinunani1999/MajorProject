import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
def predict_text(text):
    ss = sid.polarity_scores(text)
    for key in ss:
        ss[key] *= 100
    data = list()
    for key in ss:
        item = dict()
        item['sentiment'] = key
        item['value'] = int(ss[key])
        data.append(item)

    if ss['compound'] >= 5:
        result = "POSITIVE"
    elif ss['compound'] <= -5:        
        result = "NEGATIVE"
    else:
        result = "NEUTRAL"
    return data, result
