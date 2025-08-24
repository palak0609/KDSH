## Imports and Initialization
## This section imports necessary libraries and initializes key components for our paper prediction pipeline:
## pip install -U langchain-community

import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tempfile
import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import plotly.graph_objects as go
import re
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    return tokenizer, model

# Labeled papers
labeled_papers = [
    ('R006.pdf', 'CVPR'),
    ('R007.pdf', 'CVPR'),
    ('R008.pdf', 'EMNLP'),
    ('R009.pdf', 'EMNLP'),
    ('R010.pdf', 'KDD'),
    ('R011.pdf', 'KDD'),
    ('R012.pdf', 'NeurIPS'),
    ('R013.pdf', 'NeurIPS'),
    ('R014.pdf', 'TMLR'),
    ('R015.pdf', 'TMLR')
]

def get_embedding(text, tokenizer, model):
    """Generate embedding for given text"""
    max_length = 512
    text = ' '.join(text.split()[:max_length])
    
    inputs = tokenizer(text, padding=True, truncation=True, 
                      max_length=512, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def extract_paper_features(text):
    """Extract key features from paper text"""
    features = {
        'deep_learning': len(re.findall(r'\b(deep learning|neural network|CNN|RNN|LSTM|deep neural|artificial neural|convolutional|recurrent neural)\b', text.lower())),
        'computer_vision': len(re.findall(r'\b(computer vision|image processing|object detection|segmentation|visual recognition|image classification|feature detection)\b', text.lower())),
        'nlp': len(re.findall(r'\b(natural language|nlp|text mining|language model|transformer|bert|gpt|word embedding|tokenization)\b', text.lower())),
        'data_mining': len(re.findall(r'\b(data mining|clustering|pattern recognition|kdd|knowledge discovery|association rules|anomaly detection)\b', text.lower())),
        'theory': len(re.findall(r'\b(theorem|proof|lemma|theoretical|convergence|mathematical model|algorithm complexity)\b', text.lower()))
    }
    return features

def create_conference_embeddings(labeled_papers, tokenizer, model):
    """Create embeddings for each conference's papers"""
    conference_papers = {}
    conference_features = {}
    
    for filename, conference in labeled_papers:
        if conference not in conference_papers:
            conference_papers[conference] = []
            conference_features[conference] = []
        
        try:
            loader = PyPDFLoader(filename)
            pages = loader.load()
            content = ' '.join([page.page_content for page in pages[:3]])
            
            conference_papers[conference].append(content)
            conference_features[conference].append(extract_paper_features(content))
            
        except Exception as e:
            st.warning(f"Error loading {filename}: {str(e)}")
            continue
    
    conference_embeddings = {}
    for conference, papers in conference_papers.items():
        paper_embeddings = [get_embedding(paper, tokenizer, model) for paper in papers]
        conference_embeddings[conference] = paper_embeddings
    
    return conference_embeddings, conference_features

def recommend_conference(new_paper_content, conference_embeddings, conference_features, tokenizer, model):
    """Recommend conference based on content similarity and features"""
    new_paper_embedding = get_embedding(new_paper_content, tokenizer, model)
    new_paper_features = extract_paper_features(new_paper_content)
    
    conference_scores = {}
    feature_similarities = {}
    
    for conference, paper_embeddings in conference_embeddings.items():
        similarities = []
        for paper_embedding in paper_embeddings:
            similarity = cosine_similarity([new_paper_embedding], [paper_embedding])[0][0]
            similarities.append(similarity)
        
        top_similarities = sorted(similarities, reverse=True)[:2]
        conference_scores[conference] = np.mean(top_similarities)
        
        conf_features = conference_features[conference]
        feature_matches = []
        for paper_feat in conf_features:
            total_features = sum(paper_feat.values()) + sum(new_paper_features.values())
            if total_features == 0:
                match_score = 0
            else:
                common_features = sum(min(paper_feat[k], new_paper_features[k]) for k in paper_feat)
                match_score = 2 * common_features / total_features
            feature_matches.append(match_score)
        feature_similarities[conference] = np.mean(feature_matches)
    
    final_scores = {
        conf: 0.7 * emb_score + 0.3 * feature_similarities[conf]
        for conf, emb_score in conference_scores.items()
    }
    
    return max(final_scores.items(), key=lambda x: x[1]), final_scores
def get_llm_justification(paper_content, predicted_conference, similarity_score):
    """Get a detailed justification from LLM for the predicted conference"""
    
    llm = Ollama(model="llama2", temperature=0.7)
    
    justification_prompt = PromptTemplate(
        template="""Based on the analysis of this research paper, provide a detailed academic justification (50-100 words) for why it is most suitable for the {conference} conference.

        Paper content: {paper_content}
        
        Consider the following aspects specific to {conference}:
        
        CVPR: Focus on computer vision, visual computing, pattern recognition, and deep learning for vision
        NeurIPS: Emphasis on machine learning theory, neural computation, and AI algorithms
        EMNLP: Specialization in natural language processing, computational linguistics, and language understanding
        TMLR: Focus on machine learning research, methodology, and theoretical foundations
        KDD: Expertise in data mining, knowledge discovery, and data science applications
        
        Provide a formal justification that highlights:
        1. The paper's specific methodological alignment with {conference}
        2. Technical depth and theoretical foundations
        3. Experimental approach and validation methods
        4. Relevance to the conference's core focus areas
        
        Write a cohesive, academic-style justification strictly upto 100 words:""",
        input_variables=["paper_content", "conference", "similarity_score"]
    )
    
    justification_chain = LLMChain(llm=llm, prompt=justification_prompt)
    response = justification_chain.invoke({
        "paper_content": paper_content[:3000],  # Limit content length
        "conference": predicted_conference,
        "similarity_score": f"{similarity_score:.2f}"
    })
    
    return response['text'].strip()

def create_similarity_chart(scores):
    """Create interactive similarity chart"""
    df = pd.DataFrame({
        'Conference': list(scores.keys()),
        'Similarity Score': list(scores.values())
    }).sort_values('Similarity Score', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Similarity Score'],
        y=df['Conference'],
        orientation='h'
    ))
    
    fig.update_layout(
        title="Conference Similarity Scores",
        xaxis_title="Similarity Score",
        yaxis_title="Conference",
        height=400
    )
    
    return fig

def load_pdf(file):
    """Load and process PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    
    os.unlink(temp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(docs)

def main():
    st.title("Conference Recommendation for Research Paper")
    st.sidebar.header("Upload and Analyze")
    
    # Load model and tokenizer
    tokenizer, model = load_model()
    
    uploaded_file = st.sidebar.file_uploader("Upload a Research Paper (PDF)", type="pdf")
    if uploaded_file is not None:
        with st.spinner("Processing the PDF..."):
            documents = load_pdf(uploaded_file)
            st.success("PDF loaded and split into chunks!")

            # Process the paper content
            paper_content = " ".join([doc.page_content for doc in documents[:3]])
            
            with st.spinner("Analyzing paper and generating recommendations..."):
                conference_embeddings, conference_features = create_conference_embeddings(
                    labeled_papers, tokenizer, model
                )
                (recommended_conference, score), all_scores = recommend_conference(
                    paper_content, conference_embeddings, conference_features, tokenizer, model
                )
            
            # Display results
            st.header("Results")
            
            # Primary recommendation
            st.subheader("Primary Recommendation")
            st.markdown(f"**Best Match:** {recommended_conference}")
            st.markdown(f"**Confidence Score:** {score:.2f}")
            
            # Get LLM Justification
            st.subheader("Recommendation Justification")
            with st.spinner("Generating detailed justification..."):
                try:
                    llm_justification = get_llm_justification(
                        paper_content, 
                        recommended_conference,
                        score
                    )
                    st.markdown(f"*{llm_justification}*")
                except Exception as e:
                    st.error(f"Error generating justification: {str(e)}")
                    st.markdown("*Fallback to similarity score based justification*")
            
            # Similarity chart
            st.subheader("Similarity Analysis")
            fig = create_similarity_chart(all_scores)
            st.plotly_chart(fig)
            
            # Feature analysis
            st.subheader("Content Analysis")
            features = extract_paper_features(paper_content)
            total_features = sum(features.values())
            if total_features > 0:
                feature_percentages = {k: (v/total_features)*100 for k, v in features.items()}
                for feature, percentage in feature_percentages.items():
                    if percentage > 0:
                        st.write(f"- {feature.replace('_', ' ').title()}: {percentage:.1f}%")
    else:
        st.write("Please upload a PDF to start!")

if __name__ == "__main__":
    main()
    
    
##Run this using the following command
#streamlit run KDSH_Task2_Stream.py
