"""
Automated Freelancer Bidding System (Sequential Loop + Personalized AI)
=======================================================================
1. Asks for a Start Project ID.
2. Loops sequentially (ID, ID+1, ID+2...).
3. Strictly respects 16 API calls per batch(using 5 for testing)
4. Uses advanced Profile Matching & Personalized AI Bidding.
"""

import requests
import time
import numpy as np
from sentence_transformers import SentenceTransformer

import csv
import os
from datetime import datetime

# --- NEW: LOGGING CONFIGURATION ---
LOG_FILE = "bidding_logs.csv"
#COUNTRIES_FILE = "countries-to-bid.csv"
def init_log_file():
    """Creates the CSV file with headers if it doesn't exist"""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Project ID", "Title", "Profile Used", "Score", "Action", "Details"])

def log_to_csv(project_id, title, profile_name, score, action, details):
    """Writes a single row to the CSV log"""
    try:
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([timestamp, project_id, title, profile_name, score, action, details])
    except Exception as e:
        print(f" Logging failed: {e}")
        

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = "Add your API Key"
PROD_TOKEN = "Add your token"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_THRESHOLD = 0.45 

# Rate Limiting Settings
API_CALLS_LIMIT = 10 ## just for testing purpose --> 16 is og
SLEEP_DURATION = 59  # Seconds to sleep after batch

# User details (Fallback)
USER_DETAILS = {}

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

EMBEDDING_MODEL = None
CACHED_BIDDER_ID = None

# ============================================================================
# INITIALIZATION & SETUP
# ============================================================================

def ensure_embedding_model():
    """Load the sentence transformer model for similarity matching"""
    global EMBEDDING_MODEL
    if EMBEDDING_MODEL is not None:
        return EMBEDDING_MODEL
    print(" Loading embedding model...")
    EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(" Embedding model loaded.")
    return EMBEDDING_MODEL

def ensure_main_user_id():
    """Get and cache the main bidder user ID"""
    global CACHED_BIDDER_ID
    if CACHED_BIDDER_ID:
        return CACHED_BIDDER_ID

    url = "https://www.freelancer.com/api/users/0.1/self/"
    headers = {"Freelancer-OAuth-V1": PROD_TOKEN}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        CACHED_BIDDER_ID = response.json()["result"]["id"]
        print(f" Main User ID: {CACHED_BIDDER_ID}")
        return CACHED_BIDDER_ID
    except Exception as e:
        print(f" Failed to get Main User ID: {e}")
        return None

def fetch_profiles_from_api():
    """
    Fetch profiles with FULL details (tagline, description) 
    so the AI can generate personalized bids.
    """
    # We fetch specifically for your user ID to ensure we get the right data
    url = (
        "https://www.freelancer.com/api/users/0.1/profiles"
        "?user_id=85338487&webapp=1&compact=true&new_errors=true&new_pools=true"
    )
    headers = {"Freelancer-OAuth-V1": PROD_TOKEN}
    
    print(" Fetching user profiles...")
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        raw_profiles = data.get('result', {}).get('profiles', [])
        profiles = []
        
        for p in raw_profiles:
            profiles.append({
                'id': int(p.get('id')),
                'name': p.get('profile_name'),
                'tagline': p.get('tagline'),
                'description': p.get('description'), # Critical for AI Persona
                'hourly_rate': p.get('hourly_rate')
            })
        gdesc = "Elevating digital experiences begins with design that understands users, communicates clearly, and reflects a brand’s true personality. Our UI/UX and Web Design services focus on exactly that—creating interfaces that are intuitive, visually refined, and engineered to support real business outcomes. With Figma at the core of our design workflow, we bring ideas to life with precision, collaboration, and speed.\n\nEvery project starts with understanding: user research, persona definitions, journey mapping, and competitor insights. These steps help us identify what users want, where they struggle, and how to craft experiences that feel natural, effortless, and satisfying. Based on these insights, we translate requirements into structured wireframes that define layouts, flows, and logic before visual aesthetics come into play.\n\nFigma enables us to design with accuracy and flexibility, allowing real-time collaboration, rapid prototyping, responsive layout creation, and scalable design systems. High-fidelity mockups, interactive prototypes, micro-interaction previews, and component libraries are all built within a unified ecosystem—keeping design consistent across every page, module, and product feature. This ensures seamless handoff to development teams and reduces ambiguity during implementation.\n\nOur design style focuses on clarity, balance, and modern visual language. Whether you need a sleek corporate site, a dynamic product interface, a high-conversion landing page, or a complete design refresh, every screen is crafted with attention to hierarchy, color psychology, typography, spacing, and accessibility. We ensure layouts adapt beautifully across all devices—desktop, tablet, and mobile—while maintaining performance-friendly structure and intuitive user flow.\n\nFunction is always prioritized alongside aesthetics. Prototypes undergo usability validation, interaction testing, and user-journey reviews to reduce friction points and optimize the path toward your goals—be it engagement, sales, sign-ups, or retention. We also provide A/B design variants, UI refinements, CRO-focused adjustments, and systematic enhancements for long-term growth.\n\nFor larger products, dashboards, or evolving platforms, we build custom Figma design systems—complete with reusable components, grids, styles, and UI guidelines—to ensure consistency and scalability as your digital ecosystem expands. This foundation makes future updates faster, cleaner, and more reliable for both designers and developers.\n\nOur UI/UX and Web Design services cover the full spectrum: research, UX strategy, wireframing, prototyping, interface design, micro-interactions, responsive layouts, branding alignment, accessibility-compliant visuals, and developer-ready asset preparation.\n\nIf your goal is to make your digital presence more impactful, user-friendly, and visually compelling, our Figma-driven design approach ensures your product stands out with confidence and purpose. Let’s craft stunning interfaces."
        profiles.append({
            'id': 0,
            'name': 'General Profile',
            'tagline': "Figma | UI/UX",
            'description': gdesc,
            'hourly_rate': 20,
            #'match_chunks': chunks  # Essential for the scoring logic
        })
        
        print(f" Found {len(profiles)} profiles")
        #print(profiles)
        return profiles
    except Exception as e:
        print(f" Failed to fetch profiles: {e}")
        return []

# ============================================================================
#  BID EVALUATION LOGIC (COUNTRIES & RATING)
# ============================================================================

def load_country_config():
    """
    Loads the approved countries list.
    Returns a dictionary: {'Country Name': {'new': bool, 'review': bool}}
    """
    countries = {}
    if not os.path.exists(COUNTRIES_FILE):
        print(f" Warning: Countries file '{COUNTRIES_FILE}' not found. Allowing ALL countries.")
        return None  # None means "No filter active"

    try:
        with open(COUNTRIES_FILE, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip Header
            for row in reader:
                
                if len(row) >= 4:
                    name = row[1].strip().lower() # Store as lowercase for matching
                    is_new_yes = row[2].strip().upper() == "YES"
                    is_review_yes = row[3].strip().upper() == "YES"
                    countries[name] = {'new': is_new_yes, 'review': is_review_yes}
        print(f" Loaded {len(countries)} country rules.")
        return countries
    except Exception as e:
        print(f" Failed to load country config: {e}")
        return None

def fetch_client_details(owner_id):
    """
    Fetches the Client's Rating and Country from the Users API.
    Costs: 1 API Call.
    """
    url = f"https://www.freelancer.com/api/users/0.1/users/{owner_id}?employer_reputation=true"
    headers = {"Freelancer-OAuth-V1": PROD_TOKEN}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get('status') == 'success':
                user = data.get('result')
                
                # Extract Rating (Overall employer rating)
                rep = user.get('employer_reputation', {}).get('entire_history', {})
                rating = float(rep.get('overall') or 0.0)
                
                # Extract Country
                country = user.get('location', {}).get('country', {}).get('name')
                
                return {'rating': rating, 'country': country}
        return None
    except Exception as e:
        print(f"    Error fetching client details: {e}")
        return None

def evaluate_bid_eligibility(client_data, country_rules):
    """
    Decides BID or NO BID based on strict rules.
    Returns: (Decision (bool), Reason (str))
    """
    if not client_data:
        return False, "Could not fetch client data"

    rating = client_data['rating']
    country_name = (client_data['country'] or "").lower()
    
    # 1. COUNTRY CHECK
   
    is_country_allowed = False
    
    if country_rules is None:
        is_country_allowed = True # No file = Allow all
    elif country_name in country_rules:
        rules = country_rules[country_name]
        # Rule: Double YES OR Review YES -> Eligible
        if rules['new'] or rules['review']:
            is_country_allowed = True
        else:
            return False, f"Country '{client_data['country']}' is Double NO"
    else:
        # Country not in excel file
        return False, f"Country '{client_data['country']}' not in approved list"

    # 2. RATING CHECK
    # Rule: Rating = 0 -> Eligible (if country allowed)
    if rating == 0:
        if is_country_allowed:
            return True, "New Client (Rating 0) + Country OK"
        else:
            return False, f"New Client but Country '{client_data['country']}' forbidden"

    # Rule: 1.0 <= Rating <= 4.49 -> NO BID ALWAYS
    if 1.0 <= rating <= 4.49:
        return False, f"Rating {rating} is between 1.0 and 4.5"

    # Rule: Rating >= 4.5 -> Eligible (if country allowed)
    if rating >= 4.5:
        if is_country_allowed:
            return True, f"High Rating ({rating}) + Country OK"
        else:
            return False, f"High Rating but Country '{client_data['country']}' forbidden"

    return False, f"Unhandled case: Rating {rating}"
# ============================================================================
# MATCHING LOGIC
# ============================================================================

def assemble_text(obj, fields):
    return " ".join(filter(None, [obj.get(f, '') for f in fields]))

def compute_profile_matches(project, profiles):
    """
    Matches project vs profiles. 
    Returns: (passing_matches, all_scored_matches)
    """
    model = ensure_embedding_model()
    
    # 1. Assemble Project Text
    p_text = assemble_text(project, ['title', 'preview_description', 'description'])
    
    # Handle case where 'jobs' is None (jobs is usually the skills , checking to compute if they are not null by any chance)
    raw_jobs = project.get('jobs') or []
    job_names = [j.get('name') for j in raw_jobs if j.get('name')]
    p_text += " " + " ".join(job_names)
    
    # Generate Project Embedding
    p_emb = model.encode(p_text, convert_to_numpy=True, normalize_embeddings=True)
    
    all_results = []
    
    
    # 2. Compare with Profiles
    for profile in profiles:
        
            prof_text = assemble_text(profile, ['name', 'tagline', 'description'])
            prof_emb = model.encode(prof_text, convert_to_numpy=True, normalize_embeddings=True)
            score = float(np.dot(p_emb, prof_emb))
        
            all_results.append({'profile': profile, 'score': score})
    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter only those passing the threshold
    passing = [m for m in all_results if m['score'] >= EMBEDDING_THRESHOLD]
    
    return passing, all_results
# ============================================================================
# AI BID GENERATION (PERSONALIZED)
# ============================================================================

def create_personalized_prompt(project, selected_profile):
    """Build AI prompt adapted to the selected profile"""
    title = project.get('title', '')
    description = project.get('description', '')
    budget = project.get('budget', {})
    currency = project.get('currency', {}).get('code', 'USD')
    
    min_b = budget.get('minimum', 0)
    max_b = budget.get('maximum', 0)
    budget_text = f"Budget: {min_b}-{max_b} {currency}" if min_b and max_b else ""

    # Use profile-specific context
    my_role = selected_profile.get('name', 'Freelancer')
    my_tagline = selected_profile.get('tagline', '')
    # Truncate bio to save tokens, but keep enough for context
    my_bio = (selected_profile.get('description') or '')[:1500] 
    signer = f"Team Mactix - {my_role}"

    return f"""
        Enter your prompt for gemini.
"""

def call_gemini_api(prompt, max_retries=3):
    """Generate bid text using Gemini API with retry logic"""
    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(1, max_retries + 1):
        try:
            # Gemini calls do NOT count towards Freelancer Rate Limit
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 503:
                time.sleep(2 * attempt)
                continue
            if response.status_code == 429:
                time.sleep(5)
                continue

            response.raise_for_status()
            result = response.json()
            
            candidate = result.get('candidates', [{}])[0]
            parts = candidate.get('content', {}).get('parts', [])
            if parts and parts[0].get('text'):
                return parts[0]['text']
            
            raise RuntimeError('AI returned no content.')

        except Exception as exc:
            print(f"       Gemini AI Error ({attempt}/{max_retries}): {exc}")
            time.sleep(2)

    return None # Return None if AI fails after retries

def generate_bid_text(project, selected_profile):
    """Generate customized bid text for a project"""
    prompt = create_personalized_prompt(project, selected_profile)
    return call_gemini_api(prompt)

# ============================================================================
# BID PLACEMENT
# ============================================================================

def place_bid(project_id, bidder_id, profile_id, text, amount):
    """Submit bid to Freelancer API"""
    bid_url = "https://www.freelancer.com/api/projects/0.1/bids/"
    params = {
        "compact": "true",
        "new_errors": "true",
        "new_pools": "true"
    }
    headers = {
        "Freelancer-OAuth-V1": PROD_TOKEN,
        "Content-Type": "application/json",
    }
    
    # Base Payload (Without profile_id initially)
    payload = {
        "project_id": int(project_id),
        "bidder_id": int(bidder_id),
        "amount": float(amount),
        "period": 7,
        "milestone_percentage": 50,
        "description": text,
        "showcases": []
    }

    # FIX: Only add profile_id if it is NOT the General Profile (0)
    if int(profile_id) != 0:
        payload["profile_id"] = int(profile_id)
        # print(f"      Debug: Bidding with Profile ID {profile_id}")
    else:
        pass
        # print(f"      Debug: Bidding with General Account (No Profile ID)")
    
    try:
        response = requests.post(bid_url, headers=headers, params=params, json=payload, timeout=30)
        return response.json(), response.status_code
    except Exception as exc:
        return {'message': f'Connection Error: {exc}'}, 500

# ============================================================================
# MAIN AUTOMATION WORKFLOW
# ============================================================================

def run_automation_loop():
    print("\n" + "="*60)
    print("♾️  FREELANCER AUTO-BIDDER ")
    print("="*60)

    # 1. Setup
    init_log_file()
    ensure_embedding_model()
    main_user_id = ensure_main_user_id()
    profiles = fetch_profiles_from_api()
    
    if not main_user_id or not profiles:
        print("Critical Setup Failed. Exiting.")
        return

    # 2. Input Start ID
    try:
        start_input = input("\n🔢 Enter the START Project ID (e.g., 40044413): ").strip()
        current_processing_id = int(start_input)
    except ValueError:
        print("Invalid ID. Exiting.")
        return

    print(f"\nStarting loop from ID: {current_processing_id}")
    country_config = load_country_config()

    consecutive_404s = 0

    # 3. Infinite Loop
    while True:
        batch_start_time = time.time()
        api_calls = 0
        print(f"\n🚀 Starting Batch (Limit {API_CALLS_LIMIT} calls)...")
        
        while api_calls < API_CALLS_LIMIT:
            
            try:
                # --- CALL 1: FETCH PROJECT ---
                url = f"https://www.freelancer.com/api/projects/0.1/projects/{current_processing_id}/?full_description=true"
                r = requests.get(url, headers={"Freelancer-OAuth-V1": PROD_TOKEN}, timeout=5)
                api_calls += 1 
                
                if r.status_code == 429:
                    print("Rate limit hit! Forcing 60s sleep...")
                    time.sleep(60)
                    api_calls = 0; batch_start_time = time.time(); continue 

                # ==========================================================
                #  LOGIC UPDATE: Handle Found vs Not Found
                # ==========================================================
                
                if r.status_code == 200:
                    consecutive_404s = 0 # FOUND: Reset gap counter
                    
                    data = r.json()
                    if data.get('status') == 'success':
                        project = data.get('result')
                        status = project.get('status', '').lower()
                        title = project.get('title', 'N/A')
                        owner_id = project.get('owner_id')

                        print(f"Checked {current_processing_id}: [{status.upper()}] {title[:30]}...")

                        if status == 'active':
                            if api_calls >= API_CALLS_LIMIT:
                                print("Call limit near. Skipping analysis for next batch.")
                                break 
                            
                            client_data = fetch_client_details(owner_id)
                            api_calls += 1
                            
                            should_bid, reason = evaluate_bid_eligibility(client_data, country_config)
                            
                            if should_bid:
                                print(f"ELIGIBLE: {reason}")
                                matches, all_scores = compute_profile_matches(project, profiles)
                                
                                for m in all_scores:
                                    log_status = "MATCH" if m['score'] >= EMBEDDING_THRESHOLD else "SKIP"
                                    log_to_csv(current_processing_id, title, m['profile']['name'], f"{m['score']:.4f}", "ANALYSIS", f"Result: {log_status}")

                                if matches:
                                    best = matches[0]
                                    print(f"Match: {best['profile']['name']} ({best['score']:.2f})")
                                    bid_text = generate_bid_text(project, best['profile'])
                                    
                                    if bid_text and api_calls < API_CALLS_LIMIT:
                                        budget_min = project.get('budget', {}).get('minimum', 0)
                                        bid_amount = max(budget_min, 20)
                                        
                                        print(f"Placing Bid... (${bid_amount})")
                                        resp, code = place_bid(current_processing_id, main_user_id, best['profile']['id'], bid_text, bid_amount)
                                        api_calls += 1
                                        
                                        if code == 200:
                                            print("      ✅ Bid Placed!")
                                            log_to_csv(current_processing_id, title, best['profile']['name'], f"{best['score']:.4f}", "BID_SUCCESS", "Placed")
                                        else:
                                            print(f"Bid Failed: {resp.get('message')}")
                                            log_to_csv(current_processing_id, title, best['profile']['name'], f"{best['score']:.4f}", "BID_ERROR", resp.get('message'))
                                    elif not bid_text:
                                        print("AI failed gen.")
                                    else:
                                        print("Limit reached before bid.")
                                        break
                                else:
                                    print("No profile match.")
                                    log_to_csv(current_processing_id, title, "NONE", "0.00", "NO_MATCH", "AI Score low")
                            else:
                                print(f"REJECTED: {reason}")
                                log_to_csv(current_processing_id, title, "NONE", "0.00", "REJECTED_RULES", reason)
                    else:
                        print(f"Data Error ID {current_processing_id}")

                else:
                    #  NOT FOUND
                    consecutive_404s += 1
                    print(f"ID {current_processing_id}: Not Found (Gap Count: {consecutive_404s})")
                    
                    # --- NEW LOGIC: 5 MINUTE WAIT ---
                    if consecutive_404s >= 5:
                        first_missing_id = current_processing_id - 4
                        print(f"   🛑 Hit 5 consecutive gaps (IDs {first_missing_id} to {current_processing_id}).")
                        print(f"   🕒 Waiting 5 minutes for new projects to appear...")
                        
                        # 1. Sleep for 5 minutes
                        time.sleep(300) 
                        
                        # 2. Rewind ID to the first missing one
                        print(f"Rewinding back to ID: {first_missing_id}")
                        current_processing_id = first_missing_id
                        
                        # 3. Reset Counters
                        consecutive_404s = 0 
                        
                        # 4. Reset Batch Timer (So we don't get negative sleep time at end of loop)
                        batch_start_time = time.time()
                        
                        # 5. Continue loop (It will re-check the 'first_missing_id' immediately)
                        continue 

            except Exception as e:
                print(f"Error: {e}")
                api_calls += 1
                current_processing_id += 1
                continue

            # Standard Increment
            current_processing_id += 1
            time.sleep(0.5)

        # --- SMART SLEEP CALCULATION (End of Batch) ---
        elapsed_time = time.time() - batch_start_time
        remaining_time = SLEEP_DURATION - elapsed_time
        
        print(f"💤 Batch limit reached ({api_calls}/{API_CALLS_LIMIT}).")
        
        if remaining_time > 0:
            print(f"   Sleeping for remaining {remaining_time:.2f} seconds...")
            time.sleep(remaining_time)
        else:
            print("   Processing took longer than 60s. Starting next batch immediately.")
        
        print(f"   Next Start ID: {current_processing_id}")
        
if __name__ == "__main__":
    try:
        run_automation_loop()
    except KeyboardInterrupt:
        print("\n Stopped by user.")
        
#load_country_config()