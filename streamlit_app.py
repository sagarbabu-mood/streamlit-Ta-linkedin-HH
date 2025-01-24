import streamlit as st
import pandas as pd
import json
import io
import time
import random
from datetime import datetime
from openai import AzureOpenAI
import logging

today_date = datetime.now().strftime('%Y-%m-%d')

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

api_key = st.secrets["azure_openai"]["api_key"]
azure_endpoint = st.secrets["azure_openai"]["azure_endpoint"]

def initialize_openai_client(api_key):
    try:
        return AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version="2024-02-01"
        )
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        logging.error(f"Error initializing OpenAI client: {e}")
        st.stop()

def summarize_persona(client, persona_of_job, p1_list, p2_list):
    summarize_persona_prompt = f"""
        You are tasked with converting a job description into a structured JSON format. Each parameter in the job description should be represented with:
        - `criteria`: A concise summary of the requirement.
        - `must_have`: The mandatory conditions or requirements (if it's not mandatory, return `NA`).
        - `broader_context`: Detailed steps or logic to evaluate the criteria.

        **Additional Requirements:**
        - Include P1 and P2 keywords in the output JSON under the `keywords` field.
        - P1 keywords: {p1_list}
        - P2 keywords: {p2_list}

        **Important Notes:**
        - If `Must Have` is mentioned as `NA`, include `"must_have": "NA"` in the JSON output.
        - If `Broader Context` is not specified, return `"broader_context": "NA"`.

        Job Description:
        {persona_of_job}

        Example Output:
        {{
            "parameters": {{
                "age": {{
                    "criteria": "Age should be less than 30 (consider 31 if other parameters match).",
                    "must_have": "Age <30. If rest of the parameters match, we can consider 31.",
                    "broader_context": "1. Candidate will mention in Resume\n2. If DOB is not mentioned, calculate from the year of graduation\n3. If those two are not mentioned, calculate from the starting year of career\n4. Else give as 0"
                }},
                "native_language": {{
                    "criteria": "Native Language or Known Language: Marathi",
                    "must_have": "Must explicitly mention Marathi in resume.",
                    "broader_context": "1. Candidate will mention in Resume\n2. If not mentioned, infer based on candidate work locations or native location\n3. Else give as missing"
                }}
            }},
            "keywords": {{
                "p1": {p1_list},
                "p2": {p2_list}
            }}
        }}

        Return the results strictly in the above JSON format without any additional text or explanations.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": summarize_persona_prompt}]
        )
        raw_content = response.choices[0].message.content
        
        # Clean JSON response
        for prefix in ["```json", "```"]:
            if raw_content.startswith(prefix):
                raw_content = raw_content[len(prefix):]
        raw_content = raw_content.strip()
        
        # Parse JSON and add keywords if not already included
        result = json.loads(raw_content)
        if "keywords" not in result:
            result["keywords"] = {
                "p1": p1_list,
                "p2": p2_list
            }
        
        return result
    
    except Exception as e:
        st.error(f"Error summarizing persona: {e}")
        logging.error(f"Error summarizing persona: {e}")
        return {}

def summarize_candidate(client, candidate_profile, summary_of_persona, max_retries=5):
    """Improved version with enhanced error handling and logging"""
    p1_keywords = summary_of_persona.get("keywords", {}).get("p1", [])
    p2_keywords = summary_of_persona.get("keywords", {}).get("p2", [])
    
    ### Updated Prompt
    prompt = f"""
        # HeadHunting Expert

        You are an expert in Headhunting with extensive experience in talent assessment and recruitment. You are provided with the **Job Requirements**, **Candidate Profile**, **P1 Keywords**, and **P2 Keywords**. Your task is to perform a precise, criteria-based evaluation of the candidate profile against the job requirements and keywords provided.

        # Inputs
        ```
        - **Job Requirements**: Contains the requirements of a particular job.
        - **Candidate Profile**: Contains the details of a candidate.
        - **P1 Keywords**: List of primary keywords critical for the job role.
        - **P2 Keywords**: List of secondary keywords that are desirable but not mandatory for the job role.
        ```

        ## Understanding the Job Parameters:
        ```
        - Go through the **Job Requirements**, **P1 Keywords**, and **P2 Keywords** to understand the parameters and keywords of the job role.
        ```

        ## Data Extraction Instructions:
        ```
        Thoroughly review the candidate's profile and extract the required details mentioned in the 'Job Requirements'.
        1. Extract all dates (like age, education, work experience, etc.) and put them in this format (YYYY-MM-DD).
        2. Identify the job roles and their durations (consider even if duration is still present).
        3. Extract the candidate's educational qualifications with completion dates (consider even if duration is still present).
        4. List all the skills mentioned in the candidate profile.
        5. Note if any of the information (like age, language, location) is not explicitly mentioned in the candidate profile, then evaluate as below:
            - If a candidate hasn't mentioned their date explicitly but provided the `date of birth`, calculate the years or based on the graduation year.
            - Extract the speaking language based on the location and work details provided.
            - If the end date of the work experience is still present, calculate the experience from the start date to `today:{today_date}` and compare the total experience against the parameter.
        ```

        ## Job Parameters Evaluations:
        ```
        - **Step 1: Persona Matching**:
        1. If the persona is not matching 100%, the candidate profile health is **low**.
            - **Stop evaluation here**. No need to check P1 or P2 Keywords.
        2. If the persona matches 100%, proceed to **Step 2**.

        - **Step 2: P1 Keywords Evaluation**:
        1. If **P1 Keywords** are **not mentioned**, proceed to **Step 3**.
        2. If **P1 Keywords** are **mentioned but not matching 100%**, the profile health is **low**.
            - **Stop evaluation here**. No need to check P2 Keywords.
        3. If **P1 Keywords** are **mentioned and matching 100%**, proceed to **Step 3**.

        - **Step 3: P2 Keywords Evaluation**:
        1. If **P2 Keywords** are **not mentioned**, the profile health is **high**.
        2. If **P2 Keywords** are **mentioned**:
            - If the match is at least 50%, the profile health is **high**.
            - If the match is less than 50%, the profile health is **medium**.
        ```

        ## Restrictions
        ```
        - Do not make any assumptions with the information provided. Only evaluate what is explicitly mentioned in the candidate's profile.
        ```

        ## Common Missing Points:
        ```
        - Mentioning a higher keyword percentage match even though the candidate has not mentioned the keyword in their profile.
        ```

        ## Output Requirements:
        ```
        - Here is an example of the JSON object format response for your reference:
            ```
            {{
                "persona_match_percentage": 100.0,
                "p1_match_percentage": 100.0,
                "p1_matched": ["skill1", "skill2", "skill3"],
                "p1_missing": [],
                "p2_match_percentage": 75.0,
                "p2_matched": ["skill4", "skill5"],
                "p2_missing": ["skill6"],
                "profile_health": "High"
            }}
            ```
        ```

        Note: Do not include any extra content in the output.
        ```
        #MandatoryResponseVerification:
        ```
        Take your time and cross verify below things correcly to avoid inaccurate responses.
        - Verify that you have correctly understand the job parameters mentioned in the **Job Requirements** provided
        - Verify you have extracted the required data mentioned in the **Job Requirements** from candidate's profile.
        - Strictly Verify that you have evaluated the persona, p1 keywords & p2 keywords, if not recheck the candidate's profile again and do the evaluation.
        ```
    
        Here are the details:

        **Job Requirements:**  
        {summary_of_persona}

        **Candidate Profile:**  
        {candidate_profile}
        
    """
    for attempt in range(max_retries):
        try:
            candidate_name = candidate_profile.get('Candidate Name', 'Unknown')
            logging.info(f"Processing {candidate_name} (Attempt {attempt+1}/{max_retries})")
            
            # Update the model name to match your Azure deployment
            response = client.chat.completions.create(
                model="gpt-4o",  # Replace with your actual Azure deployment name
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            raw_content = response.choices[0].message.content
            result = json.loads(raw_content)
            
            # Update required keys to match the new format
            required_keys = ["p1_match_percentage", "p1_matched", "p1_missing", "p2_match_percentage", "profile_health"]
            if not all(key in result for key in required_keys):
                missing = [k for k in required_keys if k not in result]
                raise ValueError(f"Missing keys in response: {missing}")
            
            logging.info(f"Successfully processed {candidate_name}")
            return result, result["profile_health"]
            
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed for {candidate_name}: {str(e)}")
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                logging.error(f"Max retries reached for {candidate_name}")
    
    return {"error": "Max retries reached", "candidate": candidate_name}, "Error"

def process_csv_in_batches(client, df, summary_of_persona, batch_size=20):
    df['Profile Categorization'] = None
    df['Profile Health'] = None
    
    total_rows = len(df)
    progress_bar = st.progress(0)
    processed_count = 0
    
    for batch_num in range((total_rows // batch_size) + 1):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_rows)
        print(batch_num,)
        for index in range(batch_start, batch_end):
            try:
                candidate_profile = df.iloc[index].to_dict()
                
                summary, health = summarize_candidate(client, candidate_profile, summary_of_persona)
                
                # Store as JSON string for CSV compatibility
                df.at[index, 'Profile Categorization'] = json.dumps(summary, ensure_ascii=False)
                df.at[index, 'Profile Health'] = health
                processed_count += 1
                
            except Exception as e:
                logging.error(f"Error processing row {index}: {str(e)}")
                df.at[index, 'Profile Health'] = "Error"
            
            # Update progress after each candidate
            progress_bar.progress(processed_count / total_rows)
    
    return df

def main():
    st.title("Talent Acquisition & Candidate Profiling Tool")
    client = initialize_openai_client(api_key)
    
    # Job Persona Configuration
    with st.expander("Job Persona Configuration", expanded=True):
        persona_of_job = st.text_area(
            "Job Description",
            value="""Parameter: Language\nMust Have: Telugu or Tamil\nBroader context for Prompt Criteria: Candidate will mention in the languages section or based on the location""",
            height=200
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            p1_keywords = st.text_input(
                "Primary Skills (P1 - Optional)",
                placeholder="Comma-separated core skills"
            )
        with col2:
            p2_keywords = st.text_input(
                "Secondary Skills (P2 - Optional)",
                placeholder="Comma-separated nice-to-have skills"
            )
        
        # Add a button to process the persona
        if st.button("Process Persona", key="process_persona"):
            if persona_of_job:
                with st.spinner("Analyzing job description..."):
                    p1_list = [kw.strip() for kw in p1_keywords.split(",") if kw.strip()] if p1_keywords else []
                    p2_list = [kw.strip() for kw in p2_keywords.split(",") if kw.strip()] if p2_keywords else []
                    
                    persona = summarize_persona(client, persona_of_job, p1_list, p2_list)
                    st.session_state.persona = persona
                    st.subheader("Generated Persona")
                    st.json(persona)
                    st.success("Persona processed successfully!")
            else:
                st.error("Please provide a job description to process the persona.")
    
    # Candidate Processing Section
    uploaded_file = st.file_uploader("Upload Candidate Profiles (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded candidate preview:")
        st.dataframe(df.head(3))
        
        if 'persona' not in st.session_state:
            st.warning("Please process the job persona first.")
        elif st.button("Start Processing Candidates", key="process_candidates"):
            with st.spinner("Processing candidates..."):
                processed_df = process_csv_in_batches(client, df, st.session_state.persona)
                st.success(f"Processed {len(processed_df)} candidates!")
                
                csv = processed_df.to_csv(index=False).encode()
                st.download_button(
                    "Download Full Results",
                    data=csv,
                    file_name="processed_candidates.csv",
                    mime="text/csv"
                )
if __name__ == "__main__":
    main()
