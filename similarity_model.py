from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class SimilarityModel:
    def __init__(self, text, doc):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.model = AutoModel.from_pretrained("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.resume = text
        self.requirement = doc

    def encode(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input, return_dict=True)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def compute_similarity(self):
        resume_emb = self.encode(self.resume)
        requirement_emb = self.encode(self.requirement)
        scores = torch.mm(resume_emb, requirement_emb.transpose(0, 1))[0].cpu().tolist()
        doc_score_pairs = list(zip(self.requirement, scores))
        doc, score = doc_score_pairs[0]
        return score
