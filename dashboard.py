import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from pathlib import Path
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
import urllib.parse


# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Constants
ASSESSMENTS_DIR = Path("assessments")
MAX_COURSES = 5
MAX_JOBS = 5

def load_latest_assessment_results(file_prefix):
    """Generic function to load latest assessment results"""
    files = list(ASSESSMENTS_DIR.glob(f"{file_prefix}_*.json"))
    if not files:
        return None
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    with open(latest_file) as f:
        return json.load(f)

def transform_mcq_data(raw_data):
    """Convert MCQ results to standardized format"""
    if not raw_data:
        return []
    
    transformed = []
    for q in raw_data.get("questions", []):
        transformed.append({
            "question": q.get("question", ""),
            "user_answer": q.get("user_answer", ""),
            "correct_answer": q.get("answer", ""),
            "is_correct": q.get("is_correct", False),
            "difficulty": q.get("difficulty", "N/A"),
            "explanation": q.get("explanation", "")
        })
    return transformed

def transform_technical_data(raw_data):
    """Transform raw technical assessment data into structured format"""
    if not raw_data:
        return []
    
    transformed = []
    for q in raw_data.get("questions", []):
        evaluation = q.get("evaluation", {})
        transformed.append({
            "question": q.get("question", ""),
            "answer": q.get("answer", ""),
            "score": evaluation.get("score", 0),
            "feedback": evaluation.get("feedback", ""),
            "difficulty": q.get("difficulty", "N/A")
        })
    return transformed

def calculate_scores(questions, assessment_type):
    """Generic score calculation for both MCQ and Technical"""
    if not questions:
        return {"correct": 0, "total": 0, "average": 0, "percentage": 0}
    
    if assessment_type == "mcq":
        correct = sum(1 for q in questions if q.get("is_correct", False))
        total = len(questions)
        percentage = (correct / total) * 100 if total > 0 else 0
        return {
            "correct": correct,
            "total": total,
            "percentage": percentage,
            "average": percentage / 10  # Normalize to 10-point scale
        }
    else:  # technical
        scores = [q.get("score", 0) for q in questions]
        average = sum(scores) / len(scores) if scores else 0
        percentage = average * 10  # Assuming score is out of 10
        return {
            "average": average,
            "percentage": percentage,
            "correct": sum(1 for score in scores if score >= 5),  # Count passing questions
            "total": len(scores)
        }

def display_question_analysis(mcq_questions, tech_questions):
    """Display detailed analysis of questions and answers"""
    st.subheader("🔍 Question Analysis")
    
    tab1, tab2 = st.tabs(["MCQ Questions", "Technical Questions"])
    
    with tab1:
        if mcq_questions:
            mcq_df = pd.DataFrame([{
                "Question": q.get("question", "N/A")[:50] + "...",
                "Your Answer": q.get("user_answer", "N/A"),
                "Correct Answer": q.get("correct_answer", "N/A"),
                "Result": "✅" if q.get("is_correct", False) else "❌",
                "Explanation": q.get("explanation", "N/A")
            } for q in mcq_questions])
            st.dataframe(mcq_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No MCQ questions available for analysis")
    
    with tab2:
        if tech_questions:
            tech_df = pd.DataFrame([{
                "Question": q.get("question", "N/A")[:50] + "...",
                "Your Answer": q.get("answer", "N/A"),
                "Score": f"{q.get('score', 0):.1f}/10",
                "Feedback": q.get('feedback', 'N/A'),
                "Difficulty": q.get("difficulty", "N/A")
            } for q in tech_questions])
            st.dataframe(tech_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No technical questions available for analysis")

def generate_ai_recommendations(domain, mcq_questions, tech_questions):
    """Generate personalized recommendations using Groq"""
    st.header("🚀 AI Recommendations")
    
    with st.spinner("Analyzing your performance..."):
        try:
            # Combine answers for context
            combined_answers = " ".join([
                f"Question: {q.get('question', '')} Answer: {q.get('user_answer' if 'user_answer' in q else 'answer', '')}"
                for q in mcq_questions + tech_questions
            ][:10])  # Limit to 10 questions to avoid token overflow
            
            prompt = f"""
            Analyze this {domain} assessment performance:
            - MCQ: {len(mcq_questions)} questions
            - Technical: {len(tech_questions)} questions
            
            Provide specific recommendations:
            1. Top 3 strengths
            2. Top 3 areas needing improvement
            3. Suggested learning path
            
            Assessment responses:
            {combined_answers}
            """
            
            response = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a technical career coach."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            if response.choices:
                st.markdown(response.choices[0].message.content)
                
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")

def show_dashboard():
    """Main dashboard view with score displays"""
    st.title("📈 Career Assessment Dashboard")
    
    # Load and transform data
    mcq_raw = load_latest_assessment_results("mcq")
    tech_raw = load_latest_assessment_results("technical")
    
    if not mcq_raw and not tech_raw:
        st.warning("No assessment data found. Complete an assessment first!")
        return
    
    mcq_questions = transform_mcq_data(mcq_raw)
    tech_questions = transform_technical_data(tech_raw)
    
    # Get metadata
    domain = mcq_raw.get("metadata", {}).get("domain") if mcq_raw else tech_raw.get("domain", "N/A")
    st.markdown(f"**Domain:** `{domain}`")
    
    # Calculate scores
    mcq_scores = calculate_scores(mcq_questions, "mcq")
    tech_scores = calculate_scores(tech_questions, "technical")
    
    # Display scores
    st.subheader("📊 Assessment Scores")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("MCQ Score", 
                f"{mcq_scores['correct']}/{mcq_scores['total']}",
                f"{mcq_scores['percentage']:.1f}%")
    with col2:
        st.metric("Technical Score", 
                f"{tech_scores['average']:.1f}/10",
                f"{tech_scores['percentage']:.1f}%")
    
    # Visual comparison
    st.plotly_chart(px.bar(
        x=["MCQ", "Technical"],
        y=[mcq_scores['percentage'], tech_scores['percentage']],
        labels={'x': 'Assessment Type', 'y': 'Score (%)'},
        title="Performance Comparison"
    ), use_container_width=True)
    
    # Display detailed sections
    display_question_analysis(mcq_questions, tech_questions)
    generate_ai_recommendations(domain, mcq_questions, tech_questions)
    

if __name__ == "__main__":
    show_dashboard()