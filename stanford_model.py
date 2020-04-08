from NLPInfrastructure.normalizer import SentenceNormalizer

from NLPInfrastructure.resources import stopWords, prepositions, postWords, Not1Gram

normalizer = SentenceNormalizer()


def normalize_text_atomic(text):
    text = normalizer.organize_text(text)
    text = normalizer.replace_urls(text, 'MyURL')
    text = normalizer.replace_emails(text)
    text = normalizer.replace_usernames(text)
    text = normalizer.replace_hashtags(text, 'MyHashtag')
    text = normalizer.edit_arabic_letters(text)
    text = normalizer.replace_phone_numbers(text)
    text = normalizer.replace_emoji(text)
    text = normalizer.replace_duplicate_punctuation(text)

    text = normalizer.replace_consecutive_spaces(text)
    return text


def remove_stopword(text):
    words = text.split(' ')
    words_filtered = []
    for w in words:
        if (w not in stopWords) and (w not in prepositions) and (w not in postWords) and (w not in Not1Gram):
            words_filtered.append(w)
    return ' '.join(words_filtered)


def normalize_texts(texts):
    return [normalize_text_atomic(text) for text in texts]


if __name__ == '__main__':
    print("Before")
    texts = ["علی #احمد به مدرسه میرود", "علی به مدرسه رفت www.google.com"]
    print(texts)

    texts = normalize_texts(texts)
    print("After")
    print(texts)

    clean_text = texts[0]

    one_grams = [t.split(' ') for t in texts]
    print("1Grams")
    print(one_grams)
