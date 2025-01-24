import streamlit as st
import pandas as pd
import json
import io
import time
import random
from openai import AzureOpenAI
import logging

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
    
    prompt = f"""
    Analyze the candidate profile against these strict rules:
    
    **Job Requirements:**
    {json.dumps(summary_of_persona, indent=2)}
    
    **Candidate Profile:**
    {json.dumps(candidate_profile, indent=2)}
    
    **Required JSON Response:**
    {{
        "persona_match_percentage": 100.0,
        "p1_matched": ["skill1", "skill2"],
        "p1_missing": ["skill3"],
        "p2_match_percentage": 75.0,
        "profile_health": "High"
    }}
    """
    
    for attempt in range(max_retries):
        try:
            # Log candidate being processed
            candidate_name = candidate_profile.get('Candidate Name', 'Unknown')
            logging.info(f"Processing {candidate_name} (Attempt {attempt+1}/{max_retries})")
            
            # API call with timeout
            response = client.chat.completions.create(
                model="gpt-4",  # Verify this matches your Azure deployment name
                messages=[{"role": "system", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=10  # Add timeout
            )
            
            # Clean and validate response
            raw_content = response.choices[0].message.content
            cleaned_content = raw_content.strip("` \n")
            
            # Parse and validate JSON
            result = json.loads(cleaned_content)
            required_keys = ["persona_match_percentage", "p1_matched", "p1_missing", "p2_match_percentage"]
            if not all(key in result for key in required_keys):
                missing = [k for k in required_keys if k not in result]
                raise ValueError(f"Missing keys: {missing}")
            
            # Calculate health status
            result["profile_health"] = "High"  # Default
            if result["persona_match_percentage"] < 100:
                result["profile_health"] = "Low"
            elif p1_keywords and result["p1_missing"]:
                result["profile_health"] = "Low"
            elif p2_keywords and result["p2_match_percentage"] < 50:
                result["profile_health"] = "Medium"
            
            logging.info(f"Successfully processed {candidate_name}")
            return result, result["profile_health"]
            
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON response for {candidate_name}\nRaw content: {raw_content}")
        except KeyError as e:
            logging.error(f"Missing key in response: {str(e)}")
        except Exception as e:
            logging.error(f"Error processing {candidate_name}: {str(e)}")
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
    
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
            value="""Parameter: Language\nValue: Telugu or Tamil\nBroader context for Prompt Criteria: Candidate will mention in the languages section or based on the location""",
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
