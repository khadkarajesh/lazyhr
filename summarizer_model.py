from transformers import pipeline


class SummaryModel:
    def __init__(self, text):
        self.text = text
        self.model = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize(self):
        summary = self.model(self.text,
                             max_length=92,
                             min_length=80,
                             do_sample=False)
        return summary[0]['summary_text']
