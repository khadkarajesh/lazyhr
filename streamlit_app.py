import streamlit as st
from pdfminer.high_level import extract_text

from similarity_model import SimilarityModel
from summarizer_model import SummaryModel

uploaded_file = st.file_uploader("Upload your resume")
text = ""
if uploaded_file is not None:
    with open("resume.pdf", "wb") as file:
        file.write(uploaded_file.getvalue())
    text = extract_text("resume.pdf")

txt_job_description = st.text_area('Job Description', '''''', height=400)


def on_btn_click():
    if len(txt_job_description) == 0 or len(text) == 0: return
    with st.spinner("Processing...."):
        model = SimilarityModel(text=text, doc=txt_job_description)
        similarity = round(model.compute_similarity() * 100)
        st.markdown(f"##### You resume matches {round(model.compute_similarity() * 100)}% with job description")
        if similarity <= 50:
            summary_model = SummaryModel(txt_job_description)
            st.markdown(f"### Suggestion\n {summary_model.summarize()}")
            st.success("Completed!")
        else:
            st.success("Completed!")


st.button("Compare", on_click=on_btn_click())
