import streamlit as st
import pandas as pd
import json
import io
import time
import random
import requests
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Retrieve Gemini API credentials from secrets.
gemini_api_key = st.secrets["gemini"]["api_key"]
gemini_endpoint = st.secrets["gemini"]["endpoint"]

def gemini_completion(prompt):
    headers = {"Content-Type": "application/json"}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(gemini_endpoint, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                candidate = data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"] and len(candidate["content"]["parts"]) > 0:
                    return candidate["content"]["parts"][0]["text"].strip()
            return ""
        else:
            logging.error(f"Gemini API error: {response.status_code} {response.text}")
            return ""
    except Exception as e:
        logging.error(f"Exception in gemini_completion: {e}")
        return ""

def summarize_persona(persona_of_job, p1_list, p2_list):
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
                    "broader_context": "1. Candidate will mention in Resume\\n2. If DOB is not mentioned, calculate from the year of graduation\\n3. If those two are not mentioned, calculate from the starting year of career\\n4. Else give as 0"
                }},
                "native_language": {{
                    "criteria": "Native Language or Known Language: Marathi",
                    "must_have": "Must explicitly mention Marathi in resume.",
                    "broader_context": "1. Candidate will mention in Resume\\n2. If not mentioned, infer based on candidate work locations or native location\\n3. Else give as missing"
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
        raw_content = gemini_completion(summarize_persona_prompt)
        # Remove markdown formatting if present.
        for prefix in ["```json", "```"]:
            if raw_content.startswith(prefix):
                raw_content = raw_content[len(prefix):]
        raw_content = raw_content.strip()
        
        # Extract the first JSON object from the raw content.
        start_index = raw_content.find("{")
        end_index = raw_content.rfind("}")
        if start_index != -1 and end_index != -1:
            json_str = raw_content[start_index:end_index+1]
        else:
            json_str = raw_content
        
        result = json.loads(json_str)
        if "keywords" not in result:
            result["keywords"] = {"p1": p1_list, "p2": p2_list}
        
        return result
    except Exception as e:
        st.error(f"Error summarizing persona: {e}")
        logging.error(f"Error summarizing persona: {e}")
        return {}

def summarize_candidate(candidate_profile, summary_of_persona, max_retries=5):
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
            candidate_name = candidate_profile.get('Candidate Name', 'Unknown')
            logging.info(f"Processing {candidate_name} (Attempt {attempt+1}/{max_retries})")
            
            raw_content = gemini_completion(prompt)
            # For debugging: print raw content
            logging.info(f"Raw response for {candidate_name}: {raw_content}")
            
            # Clean the response: remove markdown or extra text
            cleaned_content = raw_content.strip("` \n")
            # Optionally, extract valid JSON portion if extra data is present.
            start_index = cleaned_content.find("{")
            end_index = cleaned_content.rfind("}")
            if start_index != -1 and end_index != -1:
                cleaned_content = cleaned_content[start_index:end_index+1]
            
            result = json.loads(cleaned_content)
            required_keys = ["persona_match_percentage", "p1_matched", "p1_missing", "p2_match_percentage"]
            if not all(key in result for key in required_keys):
                missing = [k for k in required_keys if k not in result]
                raise ValueError(f"Missing keys: {missing}")
            
            result["profile_health"] = "High"
            if result["persona_match_percentage"] < 100:
                result["profile_health"] = "Low"
            elif p1_keywords and result["p1_missing"]:
                result["profile_health"] = "Low"
            elif p2_keywords and result["p2_match_percentage"] < 50:
                result["profile_health"] = "Medium"
            
            logging.info(f"Successfully processed {candidate_name}")
            return result, result["profile_health"]
            
        except json.JSONDecodeError as jde:
            logging.error(f"Invalid JSON response for {candidate_name}\nRaw content: {raw_content}\nError: {jde}")
        except Exception as e:
            logging.error(f"Error processing {candidate_name}: {e}")
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {delay:.1f}s...")
                time.sleep(delay)
    
    logging.error(f"Max retries reached for {candidate_name}")
    return {"error": "Max retries reached", "candidate": candidate_name}, "Error"

def process_csv_in_batches(df, summary_of_persona, batch_size=20):
    df['Profile Categorization'] = None
    df['Profile Health'] = None
    
    total_rows = len(df)
    progress_bar = st.progress(0)
    processed_count = 0
    
    for batch_num in range((total_rows // batch_size) + 1):
        batch_start = batch_num * batch_size
        batch_end = min((batch_num + 1) * batch_size, total_rows)
        for index in range(batch_start, batch_end):
            try:
                candidate_profile = df.iloc[index].to_dict()
                summary, health = summarize_candidate(candidate_profile, summary_of_persona)
                df.at[index, 'Profile Categorization'] = json.dumps(summary, ensure_ascii=False)
                df.at[index, 'Profile Health'] = health
                processed_count += 1
            except Exception as e:
                logging.error(f"Error processing row {index}: {str(e)}")
                df.at[index, 'Profile Health'] = "Error"
            progress_bar.progress(processed_count / total_rows)
    
    return df

def main():
    st.title("Talent Acquisition & Candidate Profiling Tool")
    
    # No need to initialize an OpenAI client for Gemini.
    with st.expander("Job Persona Configuration", expanded=True):
        persona_of_job = st.text_area(
            "Job Description",
            value="""Parameter: Language\nValue: Telugu or Tamil\nBroader context for Prompt Criteria: Candidate will mention in the languages section or based on the location""",
            height=200
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            p1_keywords = st.text_input("Primary Skills (P1 - Optional)", placeholder="Comma-separated core skills")
        with col2:
            p2_keywords = st.text_input("Secondary Skills (P2 - Optional)", placeholder="Comma-separated nice-to-have skills")
        
        if st.button("Process Persona", key="process_persona"):
            if persona_of_job:
                with st.spinner("Analyzing job description..."):
                    p1_list = [kw.strip() for kw in p1_keywords.split(",") if kw.strip()] if p1_keywords else []
                    p2_list = [kw.strip() for kw in p2_keywords.split(",") if kw.strip()] if p2_keywords else []
                    persona = summarize_persona(persona_of_job, p1_list, p2_list)
                    st.session_state.persona = persona
                    st.subheader("Generated Persona")
                    st.json(persona)
                    st.success("Persona processed successfully!")
            else:
                st.error("Please provide a job description to process the persona.")
    
    uploaded_file = st.file_uploader("Upload Candidate Profiles (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded candidate preview:")
        st.dataframe(df.head(3))
        
        if 'persona' not in st.session_state:
            st.warning("Please process the job persona first.")
        elif st.button("Start Processing Candidates", key="process_candidates"):
            with st.spinner("Processing candidates..."):
                processed_df = process_csv_in_batches(df, st.session_state.persona)
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
