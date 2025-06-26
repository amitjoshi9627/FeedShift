from keybert import KeyBERT


class FeedShiftKeyWords:
    def __init__(self):
        self.model = KeyBERT()

    def extract_keywords(self, doc):
        keywords = self.model.extract_keywords(
            docs=doc,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            use_mmr=True,
            diversity=0.7,
            top_n=3,
        )
        return keywords
