import json
import re
import os
import requests
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Check for mock mode
MOCK_ANALYSIS = os.getenv('MOCK_ANALYSIS', 'false').lower() == 'true'

# GROQ Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_MODEL = "llama-3.3-70b-versatile"


# MAIN FUNCTION
def generate_insights(metrics: dict) -> dict:
    """
    Analyze Rainbow Six Siege gameplay metrics using LLM.
    Includes:
    - Short video alerts
    - Low detection confidence handling
    - Behavioral performance analysis
    """

    # MOCK MODE: Return fake LLM response for testing
    if MOCK_ANALYSIS:
        logger.info("MOCK_ANALYSIS: Returning mock LLM insights")
        return {
            "performance_score": 75.0,
            "aim_stability": 70.0,
            "playstyle": "Aggressive Fragger",
            "strengths": ["Good kill count", "Map awareness"],
            "weaknesses": ["Death count could improve"],
            "recommendations": ["Focus on utility usage", "Work on crosshair placement"],
            "summary": "Mock analysis: Solid performance with room for improvement.",
            "match_insight": "Test match analysis completed successfully."
        }

    try:
        logger.info("=" * 60)
        logger.info("STARTING LLM ANALYSIS")
        logger.info("=" * 60)

        # STEP 1: COMPREHENSIVE DATA EXTRACTION
        logger.info("Extracting comprehensive gameplay data...")

        # Core combat stats
        kills = metrics.get("kills", 0)
        deaths = metrics.get("deaths", 0)
        headshots = metrics.get("headshots", 0)
        assists = metrics.get("assists", 0)

        # Objective stats
        plants = metrics.get("plants", 0)
        bombs_found = metrics.get("bombs_found", 0)

        # Utility & map knowledge
        identified_enemies = metrics.get("identified_enemies", 0)
        penetration_kills = metrics.get("penetration", 0)
        reinforcements = metrics.get("reinforced", 0)  # Fixed: video_analyzer uses 'reinforced' key
        carrier_denied = metrics.get("carrier_denied", 0)

        # Equipment destruction
        drones_destroyed = metrics.get("drones_destroyed", 0)
        cameras_destroyed = metrics.get("cameras_destroyed", 0)

        # Match context
        duration = metrics.get("duration_seconds", 0)  # Note: video_analyzer doesn't provide this currently
        match_result = metrics.get("match_result", "Unknown")

        # Advanced metrics
        kda_ratio = metrics.get("kda_ratio", 0)
        headshot_pct = metrics.get("headshot_percentage", 0)

        logger.info(f"Core Stats - K:{kills} D:{deaths} HS:{headshots} A:{assists}")
        logger.info(f"Objective - Plants:{plants} Bombs:{bombs_found} Result:{match_result}")
        logger.info(f"Utility - Enemies ID'd:{identified_enemies} Penetration:{penetration_kills} Reinforced:{reinforcements}")
        logger.info(f"Equipment - Drones:{drones_destroyed} Cams:{cameras_destroyed} Carrier Denied:{carrier_denied}")
        logger.info(f"Advanced - KDA:{kda_ratio} HS%:{headshot_pct}%")


        # STEP 2: HARD GUARD — NO GAMEPLAY DETECTED

        if kills == 0 and deaths == 0 and plants == 0 and bombs_found == 0:
            logger.warning("⚠ No gameplay events detected — low confidence video")

            return {
                "strengths": [
                    "Video was successfully uploaded and processed."
                ],
                "weaknesses": [
                    "No clear gameplay events were detected in this clip.",
                    "Kill feed, HUD, or round timer may not have been visible."
                ],
                "coaching_tips": [
                    "Upload a clearer video with visible HUD elements.",
                    "Use at least 1080p resolution for accurate gameplay analysis.",
                    "Avoid spectator mode or cropped gameplay footage."
                ]
            }


        # STEP 3: PREPARE PROMPT WITH BEHAVIORAL CONTEXT
        logger.info("Preparing LLM prompt with behavioral context...")

        # Extract operator/weapon context
        operators_data = metrics.get("operators", {})
        operators = operators_data.get("names", []) if isinstance(operators_data, dict) else []
        weapons = metrics.get("weapons", [])
        abilities = metrics.get("unique_abilities", [])

        logger.info(f"Context - Operators:{len(operators)} Weapons:{len(weapons)} Abilities:{len(abilities)}")

        # Create generalized performance descriptors (fuzzy language)
        kill_descriptor = "no" if kills == 0 else "minimal" if kills <= 2 else "moderate" if kills <= 5 else "high" if kills <= 10 else "very high" if kills <= 15 else "exceptional"
        death_descriptor = "flawless" if deaths == 0 else "minimal" if deaths <= 2 else "moderate" if deaths <= 4 else "high"
        headshot_descriptor = "no" if headshots == 0 else "occasional" if headshot_pct < 20 else "consistent" if headshot_pct < 40 else "frequent" if headshot_pct < 60 else "elite"

        # Objective play descriptor
        objective_descriptor = "no" if (plants + bombs_found) == 0 else "limited" if (plants + bombs_found) <= 1 else "active" if (plants + bombs_found) <= 3 else "decisive"

        # Utility usage descriptor (information gathering + denial)
        utility_total = identified_enemies + drones_destroyed + cameras_destroyed + carrier_denied
        utility_descriptor = "limited" if utility_total <= 2 else "moderate" if utility_total <= 6 else "strong" if utility_total <= 10 else "excellent"

        # Map knowledge descriptor (penetration kills + reinforcements)
        map_knowledge_descriptor = "basic" if (penetration_kills + reinforcements) <= 1 else "solid" if (penetration_kills + reinforcements) <= 4 else "advanced"

        logger.info(f"Descriptors - Kills:{kill_descriptor} Deaths:{death_descriptor} HS:{headshot_descriptor}")
        logger.info(f"             Objective:{objective_descriptor} Utility:{utility_descriptor} MapKnowledge:{map_knowledge_descriptor}")

        prompt_data = json.dumps(metrics, indent=2)
        logger.debug(f"Metrics data (first 500 chars): {prompt_data[:500]}...")

        prompt = f"""
You are an expert Rainbow Six Siege coach with deep game knowledge. Analyze this player's gameplay with tactical precision.

GAMEPLAY DATA:
{prompt_data}

========================
CRITICAL ANALYSIS RULES
========================

1. VIDEO LENGTH & QUALITY ALERTS (MANDATORY - CHECK FIRST):

   A) If video duration < 90 seconds:
      - DO NOT provide detailed tactical analysis
      - In strengths: "Video processed successfully, but duration is too short for comprehensive analysis."
      - In weaknesses: Keep minimal (1-2 items max) or omit entirely
      - In coaching_tips: "Upload a longer video (2-3 minutes minimum) showing complete rounds to receive detailed performance insights and tactical recommendations."

   B) If most stats are null/zero/extremely low relative to duration:
      - This indicates detection failure, not poor play
      - In weaknesses: "Limited gameplay data captured - HUD elements may not have been clearly visible in the video."
      - In coaching_tips: "For accurate analysis, please upload a clearer video (1080p+ recommended) with visible HUD, kill feed, and operator icons. Avoid spectator mode or heavily cropped footage."

2. TIMESTAMP-AWARE PERFORMANCE JUDGMENT:
   - CRITICAL: Analyze kill/death timestamps relative to video duration
   - Examples of GOOD performance indicators:
     * 3 kills in first 30 seconds of 2-minute video = aggressive entry fragger
     * 5 kills spread across 4-minute video with no deaths = consistent pressure
     * Kills during objective time (plants/defuses) = clutch performance
   - Examples of WEAK performance indicators:
     * 1 kill in 5-minute video = passive/ineffective positioning
     * Multiple deaths in first minute = poor map awareness
     * Long gaps between engagements = hesitant playstyle
   - DO NOT judge by raw numbers alone - context matters

3. BEHAVIORAL INSIGHTS (NEVER REPEAT EXACT NUMBERS):

   ⚠️ CRITICAL: DO NOT mention specific stat numbers in strengths/weaknesses/tips

   ❌ FORBIDDEN EXAMPLES:
   - "You got 14 kills and 0 deaths"
   - "Your 7 headshots show good aim"
   - "You had 50% headshot rate"
   - "5 enemy identifications"
   - "2 penetration kills"

   ✅ CORRECT EXAMPLES (USE FUZZY DESCRIPTORS):
   - Instead of "14 kills": → "Exceptional combat effectiveness throughout the match"
   - Instead of "0 deaths": → "Flawless survival instincts and strong positioning discipline"
   - Instead of "7 headshots": → "Consistent headshot accuracy under combat pressure"
   - Instead of "50% headshot rate": → "Elite crosshair placement and precision aiming"
   - Instead of "5 enemy IDs": → "Strong reconnaissance and information gathering for team"
   - Instead of "2 penetration kills": → "Creative angle usage and map knowledge demonstrated"

   USE THESE FUZZY DESCRIPTORS:
   - Kills: exceptional/very high/high/moderate/minimal/no combat impact
   - Deaths: flawless/minimal/moderate/high/frequent survival issues
   - Headshots: elite/frequent/consistent/occasional/no precision
   - Assists: strong/moderate/limited team coordination
   - Objective play: decisive/active/limited/no objective focus
   - Utility usage: excellent/strong/moderate/limited information gathering
   - Map knowledge: advanced/solid/basic angle awareness and site preparation
   - Penetration kills: creative/occasional/no wallbang usage
   - Equipment destruction: aggressive/moderate/limited utility denial
   - Reinforcements: disciplined/adequate/minimal site setup

   Translate ALL stats into behavioral insights:
   - Kills → aggression level, positioning effectiveness, trade potential
   - Deaths → map awareness, positioning mistakes, timing issues
   - Headshots → aim quality, crosshair discipline, pre-aim habits
   - Assists → team coordination, callout quality, support play
   - Plants/Defuses → objective focus, risk management, game sense
   - Enemy identification → drone usage, information gathering, team support
   - Penetration kills → map knowledge, angle awareness, creative plays
   - Reinforcements → setup discipline, role understanding, time management
   - Drone/Camera destruction → utility denial, anti-intel play
   - Carrier denied → defensive positioning, round-saving plays

4. OPERATOR & WEAPON CONTEXT:
   - If operators detected: Reference their unique abilities and expected playstyle
     * Ash/Zofia = aggressive entry, breaching
     * Jäger/Wamai = utility denial, roaming
     * Thermite/Hibana = hard breach, critical site opening
     * Doc/Rook = anchor defense, team support
   - If weapons detected: Comment on weapon choice effectiveness
     * SMG-11 high kills = strong secondary weapon mastery
     * DMR usage = methodical, angle-holding playstyle
   - If abilities are in 'unique_abilities' list: Mention ability usage effectiveness

5. PROPORTIONAL FEEDBACK (NO FORCED BALANCE):

   DOMINANT PERFORMANCE (KDA > 3.0, high objective play):
   - Strengths: 4-6 items (detailed praise of multiple aspects)
   - Weaknesses: 0-2 items (minor optimizations only)
   - Tips: 2-3 items (advanced techniques, consistency maintenance)

   STRONG PERFORMANCE (KDA > 1.5, solid objective play):
   - Strengths: 3-4 items
   - Weaknesses: 2-3 items
   - Tips: 3-4 items

   AVERAGE PERFORMANCE (KDA ~1.0, mixed results):
   - Strengths: 2-3 items
   - Weaknesses: 3-4 items
   - Tips: 4-5 items

   WEAK PERFORMANCE (KDA < 0.5, minimal impact):
   - Strengths: 1-2 items (acknowledge any positive moments)
   - Weaknesses: 4-5 items (constructive, not demoralizing)
   - Tips: 5-6 items (fundamental improvements, practice guidance)

   ⚠️ NEVER create contradictory feedback like:
   - "Excellent aggressive entry" + "Too passive, need to push more"
   - "Great positioning" + "Poor positioning led to deaths"
   - "Strong aim" + "Work on accuracy"

6. HEADSHOT PERCENTAGE INTERPRETATION:
   - 50%+ headshot rate = Elite aim discipline
   - 30-50% = Strong aim with good crosshair placement
   - 15-30% = Average aim, room for improvement
   - <15% = Needs significant crosshair placement work
   - 0% with kills = Body shot reliance, missing critical damage

7. KDA RATIO CONTEXT:
   - >3.0 = Dominant performance
   - 2.0-3.0 = Strong impact player
   - 1.0-2.0 = Positive contribution
   - 0.5-1.0 = Struggling but contributing
   - <0.5 = Needs fundamental improvement

========================
OUTPUT FORMAT (STRICT)
========================
Return ONLY valid JSON (no markdown, no explanation):

{{
  "strengths": ["Behavioral insight about what they did well...", "..."],
  "weaknesses": ["Constructive feedback about what needs work...", "..."],
  "coaching_tips": ["Actionable advice for improvement...", "..."]
}}

REMEMBER: Be a coach, not a statistics reporter. Focus on WHY and HOW, not just WHAT happened.
"""


        # STEP 4: LLM API CALL
        logger.info(f"Calling GROQ API with model: {GROQ_MODEL}")

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,  # Slightly increased for more diverse behavioral insights
            "max_tokens": 1200  # Increased for comprehensive analysis with behavioral details
        }

        logger.debug(f"API URL: {GROQ_API_URL}")
        logger.debug(f"API Key (first 20 chars): {GROQ_API_KEY[:20]}...")

        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        logger.info(f"API Response Status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"API Error Response: {response.text}")

        response.raise_for_status()

        resp_json = response.json()
        logger.info("✓ API call successful")


        # STEP 5: EXTRACT LLM RESPONSE SAFELY
        logger.info("Extracting LLM response...")

        llm_text = None
        choices = resp_json.get("choices", [])

        if choices and isinstance(choices, list):
            message = choices[0].get("message", {})
            llm_text = message.get("content")

        if not llm_text:
            logger.error("✗ Invalid LLM response structure")
            logger.error(f"Response JSON: {resp_json}")
            raise ValueError("No LLM content")

        logger.info(f"✓ LLM response received ({len(llm_text)} characters)")
        logger.debug(f"LLM raw response: {llm_text[:200]}...")


        # STEP 6: PARSE JSON OUTPUT
        logger.info("Parsing LLM JSON output...")

        json_match = re.search(r"\{[\s\S]*\}", llm_text)

        if not json_match:
            logger.error("✗ LLM did not return valid JSON")
            logger.error(f"LLM text: {llm_text}")
            raise ValueError("LLM did not return JSON")

        parsed = json.loads(json_match.group(0))

        if not all(k in parsed for k in ("strengths", "weaknesses", "coaching_tips")):
            logger.error(f"✗ Missing expected keys. Found keys: {list(parsed.keys())}")
            raise ValueError("Missing expected keys in LLM output")

        logger.info("✓ JSON parsed successfully")
        logger.info(f"  Strengths: {len(parsed.get('strengths', []))} items")
        logger.info(f"  Weaknesses: {len(parsed.get('weaknesses', []))} items")
        logger.info(f"  Coaching tips: {len(parsed.get('coaching_tips', []))} items")
        logger.info("=" * 60)

        return parsed

    except requests.exceptions.HTTPError as e:
        logger.error(f"✗ HTTP Error during API call: {e}")
        logger.error(f"  Status Code: {e.response.status_code}")
        logger.error(f"  Response: {e.response.text}")

        return {
            "strengths": ["Video analysis completed but AI service returned an authentication error."],
            "weaknesses": ["Unable to generate AI insights due to API authentication failure."],
            "coaching_tips": ["Please contact support if this issue persists."]
        }

    except Exception as e:
        logger.exception("✗ Gameplay analysis failed")

        return {
            "strengths": ["System error during analysis."],
            "weaknesses": ["AI service failed to process gameplay."],
            "coaching_tips": ["Please try again later or re-upload the video."]
        }


