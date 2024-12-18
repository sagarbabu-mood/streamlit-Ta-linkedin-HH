import streamlit as st
import pandas as pd
import json
import io
import time
import random
from openai import AzureOpenAI
# Removed direct API key import for security
from apikey import api_key  # Ensure your API key is securely imported
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Initialize AzureOpenAI client
def initialize_openai_client(api_key):
    try:
        client = AzureOpenAI(
            azure_endpoint="https://nw-tech-wu.openai.azure.com/",
            api_key=api_key,
            api_version="2024-02-01"
        )
        return client
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        logging.error(f"Error initializing OpenAI client: {e}")
        st.stop()

# Function to summarize the persona
def summarize_persona(client, role, persona_of_job, keywords, percentages):
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
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Replace with your deployment name in Azure
            messages=messages
        )
        summary_of_persona = response.choices[0].message.content
        return summary_of_persona
    except Exception as e:
        st.error(f"Error summarizing persona: {e}")
        logging.error(f"Error summarizing persona: {e}")
        return ""

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
def summarize_candidate(client, role, candidate_profile, summary_of_persona, max_retries=3):
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
    - Calculate the percentage for each keyword list based on the provided percentages.
    - Sum the percentages for all matched keywords across the lists.
    
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
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Replace with your deployment name in Azure
                messages=messages
            )
            summary_content = response.choices[0].message.content
            profile_health = extract_profile_health(summary_content)
            return summary_content, profile_health
        except Exception as e:
            st.error(f"Error processing candidate: {e}")
            logging.error(f"Error processing candidate: {e}")
            if attempt < max_retries - 1:
                sleep_time = 2 ** attempt + random.random()
                st.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                st.error("Max retries reached. Skipping this candidate.")
                return "Error", "Unknown"

# Function to process CSV in batches
def process_csv_in_batches(client, role, summary_of_persona, df, keywords, percentages, batch_size=20):
    if 'Profile Categorization' not in df.columns:
        df['Profile Categorization'] = None
    if 'Profile Health' not in df.columns:
        df['Profile Health'] = None
    
    total_rows = len(df)
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)
    
    # Initialize progress bar and batch info display
    progress_bar = st.progress(0)
    batch_info = st.empty()

    processed_batches = []

    for batch_num in range(num_batches):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_rows)
        
        batch_data = df.iloc[batch_start:batch_end]
        batch_processed_data = batch_data.copy()

        for index, row in batch_data.iterrows():
            candidate_profile = {col: row[col] for col in batch_data.columns if col not in ['Profile Categorization', 'Profile Health']}
            summary, profile_health = summarize_candidate(client, role, candidate_profile, summary_of_persona)
            batch_processed_data.at[index, 'Profile Categorization'] = summary
            batch_processed_data.at[index, 'Profile Health'] = profile_health
        
        processed_batches.append(batch_processed_data)
        
        # Update progress bar after each batch
        progress = (batch_num + 1) / num_batches
        progress_bar.progress(progress)
        
        # Update batch processing info
        batch_info.write(f"Processed Batch {batch_num + 1}/{num_batches} ({batch_start + 1}-{batch_end})")
    
    # Concatenate all processed data
    all_processed_data = pd.concat(processed_batches)
    
    return processed_batches, all_processed_data

# Streamlit UI
def main():
    st.title("Talent Acquisition & Candidate Profiling Tool")
    
    st.sidebar.header("Configuration")
        
    client = initialize_openai_client(api_key)
    
    st.header("Job Persona Configuration")
    
    # Job Persona Input
    persona_of_job = st.text_area(
        "Job Persona",
        value="""Exp: 8-15 yrs
Location: Anywhere in India
Team handle size: 150 +
Edutech exp: At least 4 yrs
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
                value=', '.join(st.session_state.keywords[i]),
                key=f"keywords_{i+1}"
            )
            st.session_state.keywords[i] = [kw.strip() for kw in keywords_input.split(',') if kw.strip()]
            percentage_input = st.number_input(
                f"Percentage for List {i+1}",
                min_value=0, max_value=100,
                value=st.session_state.percentages[i],
                step=1,
                key=f"percentage_{i+1}"
            )
            st.session_state.percentages[i] = percentage_input
    
    # Button to add new keyword list
    if st.button("Add Keyword List"):
        add_keyword_list()
    
    # Process File Upload
    uploaded_file = st.file_uploader("Upload Candidate Profiles", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded file:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
            st.stop()
        
        # Initialize session state for processing
        if 'processed_batches' not in st.session_state:
            st.session_state.processed_batches = []
        if 'all_processed_data' not in st.session_state:
            st.session_state.all_processed_data = pd.DataFrame()
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        
        # Start processing
        if st.button("Start Processing") and not st.session_state.processing:
            st.session_state.processing = True
            with st.spinner("Processing..."):
                try:
                    summary_of_persona = summarize_persona(
                        client, 
                        "Job Title", 
                        persona_of_job, 
                        st.session_state.keywords, 
                        st.session_state.percentages
                    )
                    
                    if not summary_of_persona:
                        st.error("Failed to summarize persona. Please check your inputs and API key.")
                        st.session_state.processing = False
                        st.stop()
                    
                    processed_batches, all_processed_data = process_csv_in_batches(
                        client, 
                        "Job Title", 
                        summary_of_persona, 
                        df, 
                        st.session_state.keywords, 
                        st.session_state.percentages, 
                        batch_size=20  # Reduced batch size for better performance
                    )
                    
                    # Store in session state
                    st.session_state.processed_batches = processed_batches
                    st.session_state.all_processed_data = all_processed_data
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                finally:
                    st.session_state.processing = False
            st.success("Processing Completed!")
        
        # Display download buttons if processed_batches exist
        if st.session_state.processed_batches:
            st.header("Download Processed Batches")
            for i, batch in enumerate(st.session_state.processed_batches):
                to_write = io.BytesIO()
                try:
                    batch.to_csv(to_write, index=False)
                except Exception as e:
                    st.error(f"Error converting batch {i+1} to CSV: {e}")
                    continue
                to_write.seek(0)
                st.download_button(
                    label=f"Download Batch {i + 1}",
                    data=to_write,
                    file_name=f"processed_batch_{i + 1}.csv",
                    mime="text/csv"
                )
            
            # Provide a download button for all data
            to_write_all = io.BytesIO()
            try:
                st.session_state.all_processed_data.to_csv(to_write_all, index=False)
            except Exception as e:
                st.error(f"Error converting all processed data to CSV: {e}")
            to_write_all.seek(0)
            st.download_button(
                label="Download All Processed Profiles",
                data=to_write_all,
                file_name="processed_candidates_all.csv",
                mime="text/csv"
            )
        
        # Optional: Add a button to clear cache if needed
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")

if __name__ == "__main__":
    main()
