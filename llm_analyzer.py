import json



import re



import requests



import logging







# Configure logging



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



logger = logging.getLogger(__name__)







# GROQ Configuration



GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"



GROQ_API_KEY = None

import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")









def generate_insights(metrics: dict) -> dict:



    """



    Analyzes Rainbow Six Siege game metrics using an LLM.



    Includes a 'Guard Clause' to handle videos where no events were detected.



    """



    try:



        # --- STEP 1: Python Data Validation (The "Guard Clause") ---



        kills = metrics.get('kills', 0)



        deaths = metrics.get('deaths', 0)



        plants = metrics.get('plants', 0)



        defuses = metrics.get('defuses', 0)



        duration = metrics.get('duration_seconds', 0)







        # Logic: If 0 Kills AND 0 Deaths AND no Objective play, the detection likely failed



        # or the video quality was too poor.



        if kills == 0 and deaths == 0 and plants == 0 and defuses == 0:



            logger.warning(f"R6 Metrics empty (K:0 D:0 Dur:{duration}s). Returning low-quality video feedback.")



            return {



                "strengths": [



                    "Video received, but we couldn't detect any clear gameplay events."



                ],



                "weaknesses": [



                    "We extracted 0 Kills and 0 Deaths from this clip.",



                    "The video resolution might be too low for our AI to see the Kill Feed.",



                    "The HUD (Killfeed/Timer) might be obstructed or hidden."



                ],



                "coaching_tips": [



                    "Please upload a video with at least 720p resolution (1080p is best).",



                    "Ensure the Kill Feed (top right) and Round Timer (top center) are clearly visible.",



                    "Avoid uploading menu screens or spectator clips without UI."



                ]



            }







        # --- STEP 2: The LLM Request (Only runs if data is valid) ---



        # Flatten nested structures if necessary, but usually 'metrics' is already flat here from the analyzer



        prompt_data = json.dumps(metrics, indent=2)







        prompt = f"""



You are an expert Rainbow Six Siege coach speaking directly to a player.



The player has just finished a match with the following stats:



{prompt_data}







### ANALYSIS INSTRUCTIONS ###







1. **DATA QUALITY CHECK (CRITICAL):**



   - **Duration Check:** If 'Video Duration' is less than 60 seconds, IGNORE normal analysis. Instead, in the 'strengths' section, strictly write: "Video too short for full analysis." and in 'coaching_tips' write: "Please upload a longer video (at least 2-3 minutes) for us to provide meaningful insights."



   - **Resolution/Detection Check:** If the video is long (>60s) but 'Total Kills' and 'Total Deaths' are both 0 (or extremely low compared to duration), assume the video quality was poor. In 'weaknesses', strictly write: "Low detection confidence." and in 'coaching_tips' write: "We couldn't detect much gameplay. Please upload a higher resolution video (1080p or 1440p) to get better AI analysis results."







2. **IF DATA IS SUFFICIENT, PROVIDE:**



   - **3-6 Strengths:** Address the player as "you".



   - **3-6 Weaknesses:** Address the player as "you".



   - **4-5 Coaching Tips:** Actionable advice based on the stats (e.g., improve headshot %, die less, weapon choice).







3. **BEHAVIORAL ANALYSIS (New Section):**



   Add these 3 specific pointers to your analysis (you can weave them into strengths/weaknesses or add them as specific tips if fitting, but ensure the logic is applied):



   - **Focus:** Analyze if the player maintained a steady kill rate or had periods of inactivity.



   - **Resilience:** Did the player bounce back after deaths? Or did performance worsen later in the match (based on K/D)?



   - **Composure under Pressure:** Did they win their fights (high K/D) or lose them (high deaths)? Did they get 'clutch' medals like 'Double Kill' or 'Bloodthirsty'?







### RESPONSE FORMAT ###



Return ONLY a valid JSON object with this exact structure:



{{



  "strengths": ["..."],



  "weaknesses": ["..."],



  "coaching_tips": ["..."]



}}



"""







        headers = {



            "Authorization": f"Bearer {GROQ_API_KEY}",



            "Content-Type": "application/json"



        }







        payload = {



            "model": "llama-3.1-8b-instant",



            "messages": [{"role": "user", "content": prompt}],



            "temperature": 0.3,



            "max_tokens": 700



        }







        response = requests.post(GROQ_API_URL, headers=headers, json=payload)



        response.raise_for_status()







        # Expecting the API to return a JSON body with the assistant message in choices[0].message.content



        resp_json = response.json()



        llm_text = None







        # Robust extraction: handle both common shapes



        if isinstance(resp_json, dict):



            # common structure for chat completions



            choices = resp_json.get("choices") or []



            if choices and isinstance(choices, list):



                first = choices[0]



                # support both "message": {"content": "..."} and direct "text" or "message.content"



                if isinstance(first, dict):



                    message = first.get("message")



                    if message and isinstance(message, dict):



                        llm_text = message.get("content")



                    else:



                        llm_text = first.get("text") or first.get("content")







        if not llm_text:



            logger.error("LLM response did not contain expected 'choices[0].message.content'. Full response: %s", resp_json)



            return {



                "strengths": ["Analysis generated, but format was invalid."],



                "weaknesses": ["Could not parse AI response."],



                "coaching_tips": ["Please try analyzing the video again."]



            }







        logger.info(f"LLM raw response: {llm_text}")







        # Safely extract JSON object embedded in the LLM output (best-effort)



        json_match = re.search(r'\{[\s\S]*\}', llm_text)



        if not json_match:



            # Fallback if LLM fails to output JSON



            return {



                "strengths": ["Analysis generated, but format was invalid."],



                "weaknesses": ["Could not parse AI response."],



                "coaching_tips": ["Please try analyzing the video again."]



            }







        parsed = json.loads(json_match.group(0))



        # Ensure returned structure contains required keys (basic validation)



        if not all(k in parsed for k in ("strengths", "weaknesses", "coaching_tips")):



            logger.warning("Parsed JSON missing expected keys: %s", parsed.keys())



            return {



                "strengths": ["Analysis generated but keys missing."],



                "weaknesses": ["LLM did not return the exact expected JSON structure."],



                "coaching_tips": ["Please try analyzing the video again."]



            }







        return parsed







    except Exception as e:



        logger.exception("R6 LLM analysis failed")



        # Fallback for API errors so the pipeline doesn't crash



        return {



            "strengths": ["System Error"],



            "weaknesses": ["LLM Service Unavailable"],



            "coaching_tips": ["Please check your API keys and try again."]



        }











# Example usage



if __name__ == "__main__":



    print(json.dumps(insights, indent=2))



