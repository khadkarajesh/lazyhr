from similarity_model import SimilarityModel

model = SimilarityModel(text="I am rajesh", doc=['Hi I am not rajesh'])
print(model.compute_similarity() * 100)
