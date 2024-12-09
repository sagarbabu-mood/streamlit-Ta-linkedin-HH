# streamlit_app.py

import streamlit as st
from openai import AzureOpenAI
import pandas as pd
from apikey import api_key
import json
import io

def initialize_openai_client(api_key):
    client = AzureOpenAI(
        azure_endpoint="https://nw-tech-wu.openai.azure.com/",
        api_key=api_key,
        api_version="2024-02-01"
    )
    return client
    
# Function to summarize the persona
def summarize_persona(client, role, persona_of_job, keywords, percentages):
    # Combine keywords with their percentages
    keyword_sections = []
    for idx, keyword_list in enumerate(keywords):
        keyword_str = ', '.join(keyword_list)
        keyword_sections.append(f"List {idx+1}: {keyword_str}")
    keywords_combined = '; '.join(keyword_sections)

    summarize_persona_prompt = f"""
    {role}
    Please summarize the following job persona:
    {persona_of_job}
    
    The Keywords/Skills are:
    - Keywords: {keywords_combined}
    
    Additionally, please calculate the percentage of matching skills based on the following criteria:
    - Each keyword list has a corresponding percentage as provided.
    
    Summarize the above job persona, focusing on key skills, qualifications, experience level, and the ideal candidate’s responsibilities. Output the summary in object format with the following structure:
    {{
      "persona_name": "Job Title",
      "education_requirements": "Required educational background",
      "experience_requirements": "Required work experience and expertise",
      "required_skills": "List of key skills and technical competencies",
      "persona_summary": "Overall summary of the job persona, outlining the expectations for the candidate",
    }}
    """
    
    messages = [{"role": "system", "content": summarize_persona_prompt}]
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Replace with your deployment name in Azure
        messages=messages
    )
    
    summary_of_persona = response.choices[0].message.content
    return summary_of_persona

# Function to extract profile health
def extract_profile_health(summary_content):
    if "High" in summary_content:
        return "High"
    elif "Medium" in summary_content:
        return "Medium"
    elif "Low" in summary_content:
        return "Low"
    else:
        return "Unknown"

# Function to summarize candidate
def summarize_candidate(client, role, candidate_profile, summary_of_persona):
    summarize_candidate_profile = f"""
    {role}
    Please summarize the following candidate profile:
    {json.dumps(candidate_profile, indent=2)}
    
    The summary of persona is: {summary_of_persona}
    
    In your summary, focus on the following details:
    1. A brief overview of the candidate’s educational background (Degree, university, field of study, year of graduation).
    2. A summary of the candidate's work experience (job titles, companies, job responsibilities, and technologies used).
    3. A summary of the candidate's key skills and their industry expertise.
    4. Mention the candidate's location, and the latest job/company they worked at.
    5. Candidate's headline or professional summary (if available).
    
    Additionally, please calculate the percentage of matching skills based on the following criteria:
    - 100% for skills from the Keywords list
    
    Also, return the following data in an object format: you must follow this format
    {{
        "skills_match_percentage": "<Percentage of candidate's skills matching the job persona>",
        "keywords_matched_count": "<Total number of matched keywords>",
        "matched_keywords": ["<List of keywords that matched>"],
        "profile_health": "<How well the candidate’s profile aligns with the job persona, e.g., 'High/Medium/Low'>",
        "stability_of_candidate": {{
            "average_working_time": "<Average number of months/years spent in past roles>",
            "experience_in_latest_company": "<Time spent in the latest role, in months/years>"
        }}
    }}
    """
    
    messages = [{"role": "system", "content": summarize_candidate_profile}]
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Replace with your deployment name in Azure
        messages=messages
    )
    
    summary_content = response.choices[0].message.content
    profile_health = extract_profile_health(summary_content)
    
    return summary_content, profile_health

# Function to process CSV
def process_csv(client, role, summary_of_persona, df, keywords):
    if 'Profile Categorization' not in df.columns:
        df['Profile Categorization'] = None
    if 'Profile Health' not in df.columns:
        df['Profile Health'] = None
    
    for index, row in df.iterrows():
        candidate_profile = {col: row[col] for col in df.columns if col not in ['Profile Categorization', 'Profile Health']}
        summary, profile_health = summarize_candidate(client, role, candidate_profile, summary_of_persona)
        df.at[index, 'Profile Categorization'] = summary
        df.at[index, 'Profile Health'] = profile_health
    
    return df

# Streamlit UI
def main():
    st.title("Talent Acquisition & Candidate Profiling Tool")
    
    st.sidebar.header("Configuration")
    
    # API Key Input
    api_key = "fce9b34907b848a6902e5c37ddfc8512"
    
    if not api_key:
        st.warning("Please enter your Azure OpenAI API key to proceed.")
        st.stop()
    
    client = initialize_openai_client(api_key)
    
    st.header("Job Persona Configuration")
    
    # Job Persona Input
    persona_of_job = st.text_area(
        "Job Persona",
        value="""Exp: 8-15 yrs
Location: Any where in India
Team handle size: 150 +
Edutech exp: Atleast 4 yrs
Language: English, Hindi"""
    )
    
    st.subheader("Keywords and Percentages")
    
    # Initialize session state for dynamic keyword lists
    if 'keywords' not in st.session_state:
        st.session_state.keywords = [[]]
    if 'percentages' not in st.session_state:
        st.session_state.percentages = [100]
    
    # Function to add a new keyword list
    def add_keyword_list():
        st.session_state.keywords.append([])
        st.session_state.percentages.append(100)
    
    # Display keyword lists
    for i in range(len(st.session_state.keywords)):
        with st.expander(f"Keyword List {i+1}"):
            keywords_input = st.text_area(
                f"Keywords for List {i+1} (comma separated)",
                value=','.join(st.session_state.keywords[i]),
                key=f'keywords_{i}'
            )
            st.session_state.keywords[i] = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
            percentage = st.number_input(
                f"Percentage for List {i+1}",
                min_value=0,
                max_value=100,
                value=st.session_state.percentages[i],
                step=1,
                key=f'percentages_{i}'
            )
            st.session_state.percentages[i] = percentage
    
    if st.button("Add Another Keyword List"):
        add_keyword_list()
    
    st.subheader("Upload Candidate Profiles CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded CSV Preview:")
        st.dataframe(df.head())
        
        if st.button("Process Profiles"):
            with st.spinner("Processing..."):
                # Prepare keywords and percentages
                keywords = st.session_state.keywords
                percentages = st.session_state.percentages
                
                # Define role
                role = """
                You are a Talent Acquisition Specialist with extensive experience in headhunting and job description analysis. 
                Your role includes reviewing and matching candidates to job personas. You are tasked with:
                1. Validating whether a candidate’s degree aligns with the specified year range (2022, 2023, 2024) in the job persona.
                2. Identifying and matching keywords strictly from the provided TEACHING_LIST & DSA_LIST against a candidate’s profile and education/skills.
                3. Ensuring that only valid keywords and qualifications that meet the job persona’s requirements are considered in the summary.
                """
                
                # Summarize persona
                summary_of_persona = summarize_persona(client, role, persona_of_job, keywords, percentages)
                
                # Display persona summary
                st.subheader("Job Persona Summary")
                st.json(json.loads(summary_of_persona))
                
                # Process CSV
                processed_df = process_csv(client, role, summary_of_persona, df, keywords)
                
                # Provide download link
                to_write = io.BytesIO()
                processed_df.to_csv(to_write, index=False)
                to_write.seek(0)
                st.download_button(
                    label="Download Processed CSV",
                    data=to_write,
                    file_name="processed_candidates.csv",
                    mime="text/csv"
                )
                
                st.success("Processing completed successfully!")

if __name__ == "__main__":
    main()
