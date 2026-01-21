<<<<<<< HEAD
import json







def format_final_json(video_url, metrics, llm_result):



    kills = metrics.get('kills', 0)



    deaths = metrics.get('deaths', 0)



    assists = metrics.get('assists', 0)



    plants = metrics.get('plants', 0)



    gadgets = metrics.get('gadgets_deployed', 0)



    headshots = metrics.get('headshots', 0)



    



    # Headshot %



    hs_percentage = 0.0



    if kills > 0:



        hs_percentage = round((headshots / kills) * 100, 2)







    # KDA String



    kda_string = f"{kills}/{deaths}/{assists}"







    # Performance Score Formula (Fixed Scaling)



    # 7 kills * 15 = 105. 1 plant * 30 = 30. Total ~135.



    # 135 / 600 * 100 = 22.5 (Realistic for a short clip)



    base_score = (kills * 15) + (plants * 30) + (gadgets * 5) - (deaths * 10) + (metrics.get('bombs_found', 0) * 5)



    base_score = max(0.0, float(base_score))



    



    # Assuming a full match "Good" score is 600.



    normalized_score = min(100.0, round((base_score / 600.0) * 100, 2))







    # Aim Stability



    if kills > 0:



        aim_stability = 40.0 + (hs_percentage * 0.6)



        aim_stability = min(98.0, round(aim_stability, 2))



    else:



        aim_stability = 0.0







    final_output = {



        "metrics": {



            "match_info": {



                "match_type": "standard",



                "round_number": metrics.get('round_number', 1),



                "match_result": metrics.get('match_result', "Unknown"),



                "operator_side": metrics.get('operator_side', "Unknown")



            },



            "operators": { "names": metrics.get('operators_detected', []) },



            "agents": metrics.get('operators_detected', []),



            "performance_score": normalized_score,



            "KDA": kda_string,



            "ID_enemy": metrics.get('identified_enemies', 0),



            "headshot_percentage": hs_percentage,



            "bomb_found": metrics.get('bombs_found', 0),



            "drone_destroyed": metrics.get('drones_destroyed', 0),



            "camera_destroyed": metrics.get('cameras_destroyed', 0),



            "defuser_planted": plants,



            "aim_stability": aim_stability,



            "weapons": metrics.get('weapons', []),



            "unique_abilities": metrics.get('unique_abilities', []),



            "in_abilities": metrics.get('in_abilities', []),







            "kills": kills,



            "deaths": deaths,



            "headshots": headshots,



            "plant defuse": plants,



            "deploy gadget": gadgets,



            "Revive": metrics.get('revives', 0),



            "Armor_pack": metrics.get('armor_packs', 0),



            "Drop_plant": metrics.get('drop_plant', 0),



            "Pick_up_action": metrics.get('pick_up_action', 0),



            "sabotage": metrics.get('sabotage', 0),



            "bombs_found_count": metrics.get('bombs_found', 0),



            "count_camera_destroy": metrics.get('cameras_destroyed', 0),



            "count_dron_destroy": metrics.get('drones_destroyed', 0),



            "Round_Win_points": 1 if metrics.get('match_result') == "WIN" else 0,



            "Match_Win_points": 0,



            "assist": assists,



            "End_of_Match_Score": normalized_score,



            "Identified_Enemies": metrics.get('identified_enemies', 0),



            "gadget_destroyed_count": metrics.get('gadgets_destroyed', 0),



            "hit_percentage": None,



            



            "quick_reflexes": { "reaction_time_to_enemy_sight_ms": None, "reaction_time_to_sound_cue_ms": None },



            "clutch_performance": { "clutch_rounds_won": 0, "clutch_opportunities": 0 },



            "utility_usage": {



                "primary_gadget_uses": 0, "secondary_gadget_uses": 0, "gadget_effectiveness_score": None,



                "spotting_enemies": { "drones_used": 0, "camera_view_count": 0, "pings_made": 0, "enemy_detection_delay_ms": None }



            }



        },



        "strengths": llm_result.get("strengths", []),



        "weaknesses": llm_result.get("weaknesses", []),



        "coaching_tips": llm_result.get("coaching_tips", [])



    }



=======
import json







def format_final_json(video_url, metrics, llm_result):



    kills = metrics.get('kills', 0)



    deaths = metrics.get('deaths', 0)



    assists = metrics.get('assists', 0)



    plants = metrics.get('plants', 0)



    gadgets = metrics.get('gadgets_deployed', 0)



    headshots = metrics.get('headshots', 0)



    



    # Headshot %



    hs_percentage = 0.0



    if kills > 0:



        hs_percentage = round((headshots / kills) * 100, 2)







    # KDA String



    kda_string = f"{kills}/{deaths}/{assists}"







    # Performance Score Formula (Fixed Scaling)



    # 7 kills * 15 = 105. 1 plant * 30 = 30. Total ~135.



    # 135 / 600 * 100 = 22.5 (Realistic for a short clip)



    base_score = (kills * 15) + (plants * 30) + (gadgets * 5) - (deaths * 10) + (metrics.get('bombs_found', 0) * 5)



    base_score = max(0.0, float(base_score))



    



    # Assuming a full match "Good" score is 600.



    normalized_score = min(100.0, round((base_score / 600.0) * 100, 2))







    # Aim Stability



    if kills > 0:



        aim_stability = 40.0 + (hs_percentage * 0.6)



        aim_stability = min(98.0, round(aim_stability, 2))



    else:



        aim_stability = 0.0







    final_output = {



        "metrics": {



            "match_info": {



                "match_type": "standard",



                "round_number": metrics.get('round_number', 1),



                "match_result": metrics.get('match_result', "Unknown"),



                "operator_side": metrics.get('operator_side', "Unknown")



            },



            "operators": { "names": metrics.get('operators_detected', []) },



            "agents": metrics.get('operators_detected', []),



            "performance_score": normalized_score,



            "KDA": kda_string,



            "ID_enemy": metrics.get('identified_enemies', 0),



            "headshot_percentage": hs_percentage,



            "bomb_found": metrics.get('bombs_found', 0),



            "drone_destroyed": metrics.get('drones_destroyed', 0),



            "camera_destroyed": metrics.get('cameras_destroyed', 0),



            "defuser_planted": plants,



            "aim_stability": aim_stability,



            "weapons": metrics.get('weapons', []),



            "unique_abilities": metrics.get('unique_abilities', []),



            "in_abilities": metrics.get('in_abilities', []),







            "kills": kills,



            "deaths": deaths,



            "headshots": headshots,



            "plant defuse": plants,



            "deploy gadget": gadgets,



            "Revive": metrics.get('revives', 0),



            "Armor_pack": metrics.get('armor_packs', 0),



            "Drop_plant": metrics.get('drop_plant', 0),



            "Pick_up_action": metrics.get('pick_up_action', 0),



            "sabotage": metrics.get('sabotage', 0),



            "bombs_found_count": metrics.get('bombs_found', 0),



            "count_camera_destroy": metrics.get('cameras_destroyed', 0),



            "count_dron_destroy": metrics.get('drones_destroyed', 0),



            "Round_Win_points": 1 if metrics.get('match_result') == "WIN" else 0,



            "Match_Win_points": 0,



            "assist": assists,



            "End_of_Match_Score": normalized_score,



            "Identified_Enemies": metrics.get('identified_enemies', 0),



            "gadget_destroyed_count": metrics.get('gadgets_destroyed', 0),



            "hit_percentage": None,



            



            "quick_reflexes": { "reaction_time_to_enemy_sight_ms": None, "reaction_time_to_sound_cue_ms": None },



            "clutch_performance": { "clutch_rounds_won": 0, "clutch_opportunities": 0 },



            "utility_usage": {



                "primary_gadget_uses": 0, "secondary_gadget_uses": 0, "gadget_effectiveness_score": None,



                "spotting_enemies": { "drones_used": 0, "camera_view_count": 0, "pings_made": 0, "enemy_detection_delay_ms": None }



            }



        },



        "strengths": llm_result.get("strengths", []),



        "weaknesses": llm_result.get("weaknesses", []),



        "coaching_tips": llm_result.get("coaching_tips", [])



    }



>>>>>>> 1208943 (add)
    return final_output