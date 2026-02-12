#!/usr/bin/env python3
"""Generate EXTREMELY complex & messy marketing silo dataset.
15 files, 200-400+ cols each, 5K-15K rows, dirty data, deeply interconnected relationships.

RELATIONSHIP COMPLEXITY:
- Many-to-many: customer↔campaign, campaign↔channel, customer↔product
- Hierarchical: campaign→parent_campaign, product→bundle, interaction→parent
- Circular: customer→referred_by→customer, loyalty→referred_by→loyalty
- Temporal: customer tier changes over time, campaign status transitions
- Cross-file derived: need 3+ file joins to connect entities
- Contradictory: same entity has different attributes across files
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random
import string

np.random.seed(42)
random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "marketing_silo_v2")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Shared entity pools ───
N_CUST = 5000
N_CAMP = 1000
N_PROD = 500
N_CREAT = 1500
N_INFLR = 300
N_SEGMENTS_POOL = 100  # 100 different segment IDs
N_ADGROUPS = 3000
N_AUDIENCE = 200  # audience list IDs
N_DEALS = 150  # media deals
N_AGENCIES = 50
N_TEAMS = 30

CHANNELS = ["google_ads","meta_ads","tiktok_ads","naver_ads","kakao_moment",
            "youtube","instagram","twitter_x","linkedin","email",
            "sms","push_notification","display_network","native_ad","affiliate",
            "programmatic","ott_ads","podcast_ads","influencer","offline_event"]
SEGMENTS = ["VIP_loyal","high_value","mid_tier","at_risk","dormant",
            "new_user","price_sensitive","brand_advocate","churned","reactivated",
            "seasonal","impulse_buyer","subscription","b2b_enterprise","b2b_smb",
            "gen_z","millennial","gen_x","boomer","mobile_native"]
REGIONS = ["서울","경기","부산","대구","인천","광주","대전","울산","세종",
           "강원","충북","충남","전북","전남","경북","경남","제주"]
COUNTRIES = ["KR","US","JP","CN","VN","TH","SG","DE","UK","FR","AU","CA","IN","BR","MX"]
DEVICES = ["mobile_ios","mobile_android","desktop_windows","desktop_mac",
           "tablet_ios","tablet_android","smart_tv","console","wearable","iot"]
BROWSERS = ["chrome","safari","edge","firefox","samsung_internet","whale","opera","brave","arc"]
OS_LIST = ["iOS_17","iOS_16","Android_14","Android_13","Windows_11",
           "Windows_10","macOS_14","macOS_13","Linux","ChromeOS"]
CATEGORIES_L1 = ["의류","뷰티","전자기기","식품","가구","스포츠","도서","잡화","디지털","서비스",
                 "자동차","건강","반려동물","여행","교육"]
INTERESTS = ["패션","뷰티","테크","음식","여행","스포츠","게임","육아","반려동물","인테리어",
             "자동차","금융","교육","건강","엔터테인먼트","음악","영화","독서","요리","캠핑",
             "사진","디자인","프로그래밍","투자","와인"]

dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")

# ─── Pre-generate shared IDs for cross-file relationships ───
CUST_IDS = [f"C{str(i).zfill(6)}" for i in range(1, N_CUST+1)]
CAMP_IDS = [f"CAMP-{str(i).zfill(4)}" for i in range(1, N_CAMP+1)]
PROD_IDS = [f"PRD-{i}" for i in range(1, N_PROD+1)]
CREAT_IDS = [f"CR-{i}" for i in range(1, N_CREAT+1)]
INFLR_IDS = [f"INF-{i}" for i in range(1, N_INFLR+1)]
SEG_IDS = [f"SEG-{str(i).zfill(3)}" for i in range(1, N_SEGMENTS_POOL+1)]
AG_IDS = [f"AG-{i}" for i in range(1, N_ADGROUPS+1)]
AUD_IDS = [f"AUD-{str(i).zfill(3)}" for i in range(1, N_AUDIENCE+1)]
DEAL_IDS = [f"DEAL-{i}" for i in range(1, N_DEALS+1)]
AGENCY_IDS = [f"AGENCY-{i}" for i in range(1, N_AGENCIES+1)]
TEAM_IDS = [f"TEAM-{i}" for i in range(1, N_TEAMS+1)]

# Pre-generate relationships (many-to-many)
# Campaign → multiple segments (M:N)
CAMP_SEG_MAP = {c: random.sample(SEG_IDS, random.randint(1, 8)) for c in CAMP_IDS}
# Campaign → multiple channels (M:N)
CAMP_CH_MAP = {c: random.sample(CHANNELS, random.randint(1, 7)) for c in CAMP_IDS}
# Customer → multiple segments (M:N, changes over time)
CUST_SEG_MAP = {c: random.sample(SEG_IDS, random.randint(1, 5)) for c in CUST_IDS}
# Campaign → parent campaign (hierarchy)
CAMP_PARENT = {c: random.choice(CAMP_IDS) if random.random() > 0.6 else None for c in CAMP_IDS}
# Product → bundle products (M:N self-reference)
PROD_BUNDLE = {p: random.sample(PROD_IDS, random.randint(0, 4)) for p in PROD_IDS}
# Customer → household (many customers per household)
HOUSEHOLD_IDS = [f"HH-{i}" for i in range(1, 2000)]
CUST_HH_MAP = {c: random.choice(HOUSEHOLD_IDS) for c in CUST_IDS}
# Campaign → agency (M:1 but some campaigns have multiple agencies)
CAMP_AGENCY = {c: random.sample(AGENCY_IDS, random.randint(1, 3)) for c in CAMP_IDS}
# Customer referral chain (circular)
CUST_REFERRAL = {}
for c in CUST_IDS:
    if random.random() > 0.7:
        CUST_REFERRAL[c] = random.choice(CUST_IDS)  # can be circular!
# Campaign → influencer (M:N)
CAMP_INFLR = {c: random.sample(INFLR_IDS, random.randint(0, 5)) for c in CAMP_IDS}
# Audience list → segments (M:N)
AUD_SEG_MAP = {a: random.sample(SEG_IDS, random.randint(2, 10)) for a in AUD_IDS}
# Deal → campaigns (M:N)
DEAL_CAMP_MAP = {d: random.sample(CAMP_IDS, random.randint(1, 8)) for d in DEAL_IDS}


def uid():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))


def rdates(n, fmt=None):
    d = np.random.choice(dates, n)
    if fmt:
        return [pd.Timestamp(x).strftime(fmt) for x in d]
    return [pd.Timestamp(x) for x in d]


def dirty(series, null_pct=0.08, typo_pct=0.03):
    """Inject nulls and typos."""
    arr = list(series)
    n = len(arr)
    for i in random.sample(range(n), int(n * null_pct)):
        arr[i] = random.choice([None, "", "N/A", "null", "nan", "-", "UNKNOWN", "미입력"])
    if n > 0 and arr[0] is not None and isinstance(arr[0], str):
        for i in random.sample(range(n), min(int(n * typo_pct), n)):
            if arr[i] and isinstance(arr[i], str) and len(arr[i]) > 2:
                pos = random.randint(0, len(arr[i])-1)
                arr[i] = arr[i][:pos] + random.choice("xyzqw_!") + arr[i][pos+1:]
    return arr


def mixed_dates(n):
    """Same column, DIFFERENT date formats."""
    out = []
    for _ in range(n):
        d = pd.Timestamp(np.random.choice(dates))
        fmt = random.choice(["%Y-%m-%d","%Y/%m/%d","%d-%m-%Y","%m/%d/%Y",
                             "%Y%m%d","%d.%m.%Y","%Y년 %m월 %d일","%b %d, %Y"])
        out.append(d.strftime(fmt) if random.random() > 0.05 else "")
    return out


def multi_id_col(id_list, n, max_ids=5, sep="|"):
    """Generate column with multiple IDs joined (M:N relationship indicator)."""
    return [sep.join(random.sample(id_list, random.randint(1, max_ids))) if random.random() > 0.15 else random.choice(id_list) for _ in range(n)]


def bulk_metrics(n, prefixes, suffixes=None, scale=100):
    if suffixes is None:
        suffixes = ["_raw","_norm","_pct","_delta","_yoy","_mom","_wow",
                    "_avg7d","_avg30d","_avg90d","_median","_p25","_p75","_p95",
                    "_std","_min","_max","_sum","_cnt","_weighted","_adjusted",
                    "_organic","_paid","_cumul","_rolling","_ewm"]
    data = {}
    for p in prefixes:
        for s in suffixes:
            vals = np.round(np.random.exponential(scale, n), 2)
            vals[np.random.random(n) < 0.03] = 0
            vals[np.random.random(n) < 0.01] *= 100
            data[f"{p}{s}"] = vals
    return data


def bulk_scores(n, names):
    data = {}
    for name in names:
        data[f"score_{name}"] = np.round(np.random.beta(2, 5, n) * 100, 2)
        data[f"score_{name}_prev"] = np.round(np.random.beta(2, 5, n) * 100, 2)
        data[f"score_{name}_trend"] = dirty(np.random.choice(["up","down","stable","volatile","급등","급락","횡보"], n))
        data[f"score_{name}_rank"] = np.random.randint(1, n+1, n)
        data[f"score_{name}_pctile"] = np.random.randint(0, 101, n)
    return data


# ═══════════════════════════════════════════════════════════════
# 1. customer_master.csv (~380 cols, 5000 rows)
# RELATIONSHIPS: → segments(M:N), → household(M:1), → referred_by(self),
#                → company(M:1), → preferred_products(M:N)
# ═══════════════════════════════════════════════════════════════
def gen_customers():
    print("  [1/15] customer_master.csv ...")
    n = N_CUST
    d = {}

    # Core IDs
    d["cust_id"] = CUST_IDS
    d["member_no"] = dirty([f"MEM-{random.randint(100000,999999)}" for _ in range(n)])
    d["external_uid"] = [f"ext_{uid()}" for _ in range(n)]
    d["crm_key"] = dirty([f"CRM_{random.randint(10000000,99999999)}" for _ in range(n)])
    d["hashed_email"] = [f"h_{uid()}" for _ in range(n)]
    d["hashed_phone"] = [f"p_{uid()}" for _ in range(n)]
    d["adid_gaid"] = dirty([f"gaid_{uid()}" for _ in range(n)], null_pct=0.25)
    d["idfa"] = dirty([f"idfa_{uid()}" for _ in range(n)], null_pct=0.35)

    # RELATIONSHIP: household (M:1, multiple customers share household)
    d["household_id"] = [CUST_HH_MAP[c] for c in CUST_IDS]
    d["household_role"] = dirty(np.random.choice(["primary","secondary","child","other",""], n))
    d["household_size"] = np.random.randint(1, 7, n)

    # RELATIONSHIP: referral chain (self-reference, potentially circular)
    d["referred_by_cust_id"] = [CUST_REFERRAL.get(c, "") for c in CUST_IDS]
    d["referral_depth"] = np.random.randint(0, 8, n)  # how deep in referral chain
    d["referral_code"] = [f"REF_{uid()[:6]}" if random.random() > 0.5 else "" for _ in range(n)]

    # RELATIONSHIP: segments (M:N, pipe-separated)
    d["segment_ids"] = ["|".join(CUST_SEG_MAP[c]) for c in CUST_IDS]
    d["primary_segment"] = [CUST_SEG_MAP[c][0] for c in CUST_IDS]
    d["segment_count"] = [len(CUST_SEG_MAP[c]) for c in CUST_IDS]
    d["segment_last_updated"] = mixed_dates(n)

    # RELATIONSHIP: preferred products (M:N)
    d["preferred_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=8)
    d["preferred_categories"] = multi_id_col(CATEGORIES_L1, n, max_ids=4)

    # RELATIONSHIP: company/B2B (M:1)
    d["company_id"] = dirty([f"CORP-{random.randint(1,500)}" if random.random() > 0.6 else "" for _ in range(n)], null_pct=0.3)
    d["company_contact_cust_id"] = dirty([random.choice(CUST_IDS) if random.random() > 0.8 else "" for _ in range(n)], null_pct=0.5)  # points back to another customer!

    # RELATIONSHIP: assigned team/agent
    d["assigned_team"] = dirty([random.choice(TEAM_IDS) if random.random() > 0.4 else "" for _ in range(n)])
    d["assigned_agency"] = dirty([random.choice(AGENCY_IDS) if random.random() > 0.7 else "" for _ in range(n)])

    # Demographics (30+)
    d["first_name"] = dirty(np.random.choice(["민준","서연","지훈","수빈","예준","지아","도윤","하은","시우","서윤","John","Emily","太郎","花子"], n))
    d["last_name"] = dirty(np.random.choice(["김","이","박","최","정","강","조","윤","장","임","Smith","Wang","田中","佐藤"], n))
    d["gender_code"] = dirty(np.random.choice(["M","F","O","U","male","female","남","여","기타","Male","FEMALE","m","f"], n))
    d["birth_date"] = mixed_dates(n)
    d["age"] = np.random.randint(15, 85, n)
    d["age_group"] = dirty(np.random.choice(["10대","20대","30대","40대","50대","60+","teens","20s","30s","40s","50s","60+"], n))
    d["registration_dt"] = rdates(n, "%Y/%m/%d")
    d["last_login_dt"] = mixed_dates(n)
    d["account_status"] = dirty(np.random.choice(["active","inactive","suspended","pending","deleted","ACTIVE","Active","활성","비활성","정지"], n))
    d["membership_tier"] = dirty(np.random.choice(["bronze","silver","gold","platinum","diamond","vvip","Bronze","GOLD","브론즈","실버","골드","플래티넘"], n))
    d["region_primary"] = dirty(np.random.choice(REGIONS + ["Seoul","Busan","Tokyo","New York","Shanghai"], n))
    d["country_code"] = dirty(np.random.choice(COUNTRIES + ["대한민국","미국","일본","Korea","USA","Japan"], n))
    d["city"] = dirty(np.random.choice(["서울","부산","인천","대구","Seoul","Busan","東京","NYC","上海","수원","성남"], n))
    d["postal_code"] = dirty([f"{random.randint(10000,99999)}" for _ in range(n)])
    d["timezone"] = dirty(np.random.choice(["Asia/Seoul","UTC+9","KST","America/New_York","EST","Asia/Tokyo","JST","UTC","GMT+9"], n))
    d["language_pref"] = dirty(np.random.choice(["ko","en","ja","zh","ko-KR","en-US","ja-JP","한국어","English","日本語"], n))
    d["email_domain"] = dirty(np.random.choice(["gmail.com","naver.com","daum.net","kakao.com","outlook.com","yahoo.com","hanmail.net"], n))
    d["phone_carrier"] = dirty(np.random.choice(["SKT","KT","LGU+","알뜰폰","sk텔레콤","케이티","엘지유플러스"], n))
    d["device_primary"] = dirty(np.random.choice(DEVICES, n))
    d["os_primary"] = dirty(np.random.choice(OS_LIST, n))
    d["app_installed"] = np.random.choice([1, 0, "Y", "N", "true", "false", None], n)
    d["push_opt_in"] = np.random.choice([1, 0, "Y", "N", "yes", "no", None], n)
    d["email_opt_in"] = np.random.choice([1, 0, "Y", "N", "동의", "미동의", None], n)

    # Behavioral scores (25 × 5 = 125 cols)
    score_names = ["engagement","loyalty","churn_risk","lifetime_value","purchase_freq",
                   "brand_affinity","price_sensitivity","recency","monetary","frequency",
                   "aov","basket_size","return_rate","support_tickets","nps",
                   "csat","ces","session_depth","time_on_site","bounce_rate",
                   "email_responsiveness","push_responsiveness","social_engagement",
                   "referral_propensity","discount_dependency"]
    d.update(bulk_scores(n, score_names))

    # Purchase history (6 periods × 7 = 42 cols)
    for period in ["7d","30d","90d","180d","365d","lifetime"]:
        d[f"purchases_{period}"] = np.random.poisson(3 if "life" in period else 1, n)
        d[f"revenue_{period}_krw"] = np.round(np.random.exponential(80000, n), 0)
        d[f"orders_{period}"] = np.random.poisson(2, n)
        d[f"items_{period}"] = np.random.poisson(5, n)
        d[f"returns_{period}"] = np.random.poisson(0.3, n)
        d[f"avg_order_value_{period}"] = np.round(np.random.exponential(40000, n), 0)
        d[f"unique_categories_{period}"] = np.random.randint(0, 15, n)

    # Channel preferences (20 channels)
    for ch in CHANNELS:
        d[f"ch_pref_{ch}"] = np.round(np.random.uniform(0, 1, n), 3)

    # ML predictions (12 × 3 = 36 cols)
    for model in ["propensity_buy","propensity_churn","propensity_upsell","propensity_cross_sell",
                  "propensity_reactivate","clv_predicted","next_purchase_days","category_affinity",
                  "discount_sensitivity","channel_pref_score","content_affinity","time_preference"]:
        d[f"ml_{model}"] = np.round(np.random.uniform(0, 1, n), 4)
        d[f"ml_{model}_conf"] = np.round(np.random.uniform(0.3, 1, n), 3)
        d[f"ml_{model}_ver"] = np.random.choice(["v1.0","v1.1","v2.0","v2.1","v3.0","deprecated"], n)

    # Interest affinities (25 interests)
    for interest in INTERESTS:
        d[f"interest_{interest}"] = np.round(np.random.uniform(0, 1, n), 3)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "customer_master.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 2. campaign_registry.csv (~320 cols, 1000 rows)
# RELATIONSHIPS: → parent_campaign(self), → segments(M:N), → channels(M:N),
#                → audiences(M:N), → agencies(M:N), → influencers(M:N),
#                → products(M:N), → deals(M:N), → team(M:1)
# ═══════════════════════════════════════════════════════════════
def gen_campaigns():
    print("  [2/15] campaign_registry.csv ...")
    n = N_CAMP
    d = {}

    d["camp_id"] = CAMP_IDS
    d["campaign_name"] = dirty([f"캠페인_{random.choice(['봄','여름','가을','겨울','연중'])}_{random.choice(['세일','프로모','브랜딩','리타겟','신규','CRM'])}_{i}" for i in range(1,n+1)])

    # RELATIONSHIP: parent campaign (self-reference hierarchy)
    d["parent_campaign_id"] = [CAMP_PARENT[c] if CAMP_PARENT[c] else "" for c in CAMP_IDS]
    d["campaign_hierarchy_level"] = np.random.choice([0,1,2,3], n, p=[0.4,0.3,0.2,0.1])

    # RELATIONSHIP: target segments (M:N, pipe-separated)
    d["target_segment_ids"] = ["|".join(CAMP_SEG_MAP[c]) for c in CAMP_IDS]
    d["target_segment_count"] = [len(CAMP_SEG_MAP[c]) for c in CAMP_IDS]

    # RELATIONSHIP: channels (M:N, pipe-separated)
    d["channel_ids"] = ["|".join(CAMP_CH_MAP[c]) for c in CAMP_IDS]
    d["primary_channel"] = [CAMP_CH_MAP[c][0] for c in CAMP_IDS]
    d["channel_count"] = [len(CAMP_CH_MAP[c]) for c in CAMP_IDS]

    # RELATIONSHIP: audience lists (M:N)
    d["audience_list_ids"] = multi_id_col(AUD_IDS, n, max_ids=5)
    d["exclude_audience_ids"] = multi_id_col(AUD_IDS, n, max_ids=3)

    # RELATIONSHIP: agencies (M:N)
    d["agency_ids"] = ["|".join(CAMP_AGENCY[c]) for c in CAMP_IDS]
    d["lead_agency"] = [CAMP_AGENCY[c][0] for c in CAMP_IDS]

    # RELATIONSHIP: influencers (M:N)
    d["influencer_ids"] = ["|".join(CAMP_INFLR[c]) if CAMP_INFLR[c] else "" for c in CAMP_IDS]
    d["influencer_count"] = [len(CAMP_INFLR[c]) for c in CAMP_IDS]

    # RELATIONSHIP: products (M:N)
    d["featured_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=10)
    d["excluded_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=3)

    # RELATIONSHIP: deals (M:N)
    d["media_deal_ids"] = multi_id_col(DEAL_IDS, n, max_ids=4)

    # RELATIONSHIP: team (M:1)
    d["owning_team_id"] = [random.choice(TEAM_IDS) for _ in range(n)]
    d["secondary_team_ids"] = multi_id_col(TEAM_IDS, n, max_ids=3)

    # RELATIONSHIP: related campaigns (M:N - cross-promotions)
    d["related_campaign_ids"] = multi_id_col(CAMP_IDS, n, max_ids=5)
    d["competing_campaign_ids"] = multi_id_col(CAMP_IDS, n, max_ids=3)  # cannibalization tracking

    # RELATIONSHIP: creative assets (M:N)
    d["creative_asset_ids"] = multi_id_col(CREAT_IDS, n, max_ids=15)
    d["hero_creative_id"] = [random.choice(CREAT_IDS) for _ in range(n)]

    d["campaign_type"] = dirty(np.random.choice(["awareness","consideration","conversion","retention","reactivation","branding","performance","seasonal","launch","event","always_on","guerrilla"], n))
    d["campaign_objective"] = dirty(np.random.choice(["traffic","leads","sales","app_install","video_views","engagement","reach","brand_lift","store_visit","catalog_sales"], n))
    d["start_date"] = rdates(n, "%d-%m-%Y")
    d["end_date"] = rdates(n, "%d-%m-%Y")
    d["status"] = dirty(np.random.choice(["draft","active","paused","completed","archived","cancelled","ACTIVE","Draft","진행중","완료"], n))

    # Channel-level config (20 channels × 3 = 60 cols)
    for ch in CHANNELS:
        d[f"ch_{ch}_on"] = np.random.choice([0, 1, "Y", "N", None], n, p=[0.35, 0.35, 0.1, 0.1, 0.1])
        d[f"ch_{ch}_budget_pct"] = np.round(np.random.uniform(0, 0.3, n), 3)
        d[f"ch_{ch}_priority"] = np.random.choice(["high","medium","low","critical",None], n)

    # Targeting (17 regions + 20 segments + 10 cols = ~50)
    d["target_age_min"] = np.random.choice([13,18,20,25,30,35,40], n)
    d["target_age_max"] = np.random.choice([25,35,45,55,65,99], n)
    d["target_gender"] = dirty(np.random.choice(["all","M","F","M+F","male","female","전체"], n))
    for reg in REGIONS:
        d[f"geo_{reg}"] = np.random.choice([0, 1, None], n, p=[0.6, 0.3, 0.1])

    # KPIs (20 × 4 = 80 cols)
    d["total_budget_krw"] = np.random.choice([500000,1000000,5000000,10000000,50000000,100000000], n)
    d["bid_strategy"] = dirty(np.random.choice(["manual_cpc","auto_cpc","target_cpa","target_roas","max_conversions","smart_bidding"], n))
    kpis = ["impressions","clicks","conversions","spend","revenue","ctr","cvr","cpc","cpa","roas",
            "cpm","cpv","vtr","engagement_rate","bounce_rate","session_duration","pages_per_session",
            "new_users","returning_users","assisted_conversions"]
    for kpi in kpis:
        d[f"kpi_{kpi}_target"] = np.round(np.random.exponential(1000, n), 2)
        d[f"kpi_{kpi}_actual"] = np.round(np.random.exponential(1000, n), 2)
        d[f"kpi_{kpi}_achievement"] = np.round(np.random.uniform(0.1, 3.0, n), 3)
        d[f"kpi_{kpi}_status"] = dirty(np.random.choice(["on_track","behind","ahead","at_risk","exceeded","미달","초과"], n))

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "campaign_registry.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 3. ad_impressions_log.csv (~300 cols, 15000 rows)
# RELATIONSHIPS: → campaign(M:1), → customer(M:1), → creative(M:1),
#                → adgroup(M:1), → placement/deal(M:1), → product(M:N),
#                → audience(M:N), → session_id(→web_analytics)
# ═══════════════════════════════════════════════════════════════
def gen_impressions():
    print("  [3/15] ad_impressions_log.csv ...")
    n = 15000
    d = {}

    d["impression_id"] = [f"IMP-{uid()}" for _ in range(n)]
    d["campaign_code"] = dirty([random.choice(CAMP_IDS) for _ in range(n)])
    d["user_hash"] = dirty([random.choice(CUST_IDS) if random.random() > 0.2 else f"anon_{uid()}" for _ in range(n)])
    d["creative_ref"] = dirty([random.choice(CREAT_IDS) for _ in range(n)])
    d["ad_group_id"] = dirty([random.choice(AG_IDS) for _ in range(n)])
    d["placement_id"] = [f"PL-{random.randint(1,800)}" for _ in range(n)]
    d["publisher_id"] = dirty([f"PUB-{random.randint(1,200)}" for _ in range(n)])

    # RELATIONSHIP: deal (→ deals, which → campaigns = derived relationship)
    d["deal_id"] = dirty([random.choice(DEAL_IDS) if random.random() > 0.5 else "" for _ in range(n)], null_pct=0.2)

    # RELATIONSHIP: products shown in ad (M:N)
    d["product_ids_shown"] = multi_id_col(PROD_IDS, n, max_ids=6)

    # RELATIONSHIP: audience list matched
    d["matched_audience_ids"] = multi_id_col(AUD_IDS, n, max_ids=4)
    d["matched_segment_ids"] = multi_id_col(SEG_IDS, n, max_ids=5)

    # RELATIONSHIP: web session (links to web_analytics_sessions)
    d["session_ref"] = dirty([f"SES-{uid()}" if random.random() > 0.3 else "" for _ in range(n)], null_pct=0.2)

    # RELATIONSHIP: conversion (links to conversion_funnel)
    d["conversion_ref"] = dirty([f"CVN-{uid()}" if random.random() > 0.85 else "" for _ in range(n)], null_pct=0.5)

    # Timestamps
    base_ts = int(datetime(2023, 1, 1).timestamp() * 1000)
    d["event_timestamp_ms"] = [base_ts + random.randint(0, 94608000000) for _ in range(n)]
    d["event_date_yyyymmdd"] = [datetime.fromtimestamp(t/1000).strftime("%Y%m%d") for t in d["event_timestamp_ms"]]
    d["event_hour"] = [datetime.fromtimestamp(t/1000).hour for t in d["event_timestamp_ms"]]

    # Channel & placement (20+ cols)
    d["channel"] = dirty(np.random.choice(CHANNELS, n))
    d["sub_channel"] = dirty(np.random.choice(["search","display","video","shopping","app","social_feed","stories","reels","explore","messenger","pre_roll","mid_roll","native","interstitial"], n))
    d["ad_format"] = dirty(np.random.choice(["text","image","video","carousel","native","responsive","shopping","app_install","lead_form","playable","interactive","dynamic_creative"], n))
    d["ad_size"] = dirty(np.random.choice(["300x250","728x90","160x600","320x50","1080x1080","1200x628","1080x1920","300x600","auto","unknown"], n))
    d["position"] = dirty(np.random.choice(["top","side","bottom","feed","stories","pre_roll","mid_roll","in_article","overlay","interstitial","header_bidding"], n))

    # Device & geo (25+ cols)
    d["device_type"] = dirty(np.random.choice(DEVICES, n))
    d["os_version"] = dirty(np.random.choice(OS_LIST, n))
    d["browser"] = dirty(np.random.choice(BROWSERS, n))
    d["connection"] = dirty(np.random.choice(["wifi","4g","5g","3g","wired","satellite","unknown"], n))
    d["country"] = dirty(np.random.choice(COUNTRIES, n))
    d["region"] = dirty(np.random.choice(REGIONS + ["unknown","해외"], n))
    d["latitude"] = np.where(np.random.random(n) > 0.1, np.round(np.random.uniform(33, 38.5, n), 6), None)
    d["longitude"] = np.where(np.random.random(n) > 0.1, np.round(np.random.uniform(126, 130, n), 6), None)
    d["is_vpn"] = np.random.choice([0, 1, None, "unknown"], n, p=[0.85, 0.05, 0.05, 0.05])

    # Engagement (8 × 26 = 208 cols)
    d.update(bulk_metrics(n, ["impr","click","view","video","hover","scroll","interact","engage"], scale=50))

    # Cost & bid (12+ cols)
    d["bid_amount"] = np.round(np.random.exponential(500, n), 0)
    d["win_price"] = np.round(np.array(d["bid_amount"]) * np.random.uniform(0.2, 1.0, n), 0)
    d["cost"] = np.round(np.random.exponential(300, n), 0)
    d["bid_type"] = dirty(np.random.choice(["cpc","cpm","cpv","cpa","ocpm","cpe"], n))
    d["quality_score"] = np.random.randint(1, 11, n)
    d["relevance_score"] = np.round(np.random.uniform(0, 10, n), 2)

    # Attribution (12+ cols)
    d["attr_model"] = dirty(np.random.choice(["last_click","first_click","linear","time_decay","position_based","data_driven","markov","shapley"], n))
    d["attr_weight"] = np.round(np.random.uniform(0, 1, n), 4)
    d["touchpoint_order"] = np.random.randint(1, 20, n)
    d["total_touchpoints"] = np.random.randint(1, 50, n)
    d["cross_device"] = np.random.choice([0, 1, "Y", "N", None], n)
    d["cross_channel"] = np.random.choice([0, 1, "Y", "N", None], n)

    # Brand safety (10+ cols)
    d["brand_safety_score"] = np.round(np.random.uniform(0.3, 1.0, n), 3)
    d["fraud_score"] = np.round(np.random.exponential(0.05, n), 4)
    d["ivt_flag"] = np.random.choice([0, 1, "GIVT", "SIVT", None], n, p=[0.85, 0.03, 0.05, 0.02, 0.05])
    d["brand_safety_cat"] = dirty(np.random.choice(["safe","low_risk","medium_risk","high_risk","blocked"], n))

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "ad_impressions_log.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 4. email_marketing_detail.csv (~250 cols, 10000 rows)
# RELATIONSHIPS: → customer(M:1), → campaign(M:1), → products(M:N),
#                → segment(M:N), → creative(M:1), → ab_test(M:1),
#                → conversion(M:1), → next_email(self-chain)
# ═══════════════════════════════════════════════════════════════
def gen_email():
    print("  [4/15] email_marketing_detail.csv ...")
    n = 10000
    d = {}

    d["email_event_id"] = [f"EM-{uid()}" for _ in range(n)]
    d["subscriber_id"] = dirty([random.choice(CUST_IDS) for _ in range(n)])
    d["promo_id"] = dirty([random.choice(CAMP_IDS) for _ in range(n)])
    d["email_list_id"] = dirty([random.choice(AUD_IDS) for _ in range(n)])  # → audience list!
    d["template_id"] = dirty([f"TPL-{random.randint(1,300)}" for _ in range(n)])
    d["creative_asset_id"] = dirty([random.choice(CREAT_IDS) for _ in range(n)])  # → creative
    d["send_datetime"] = rdates(n, "%m/%d/%Y %H:%M")

    # RELATIONSHIP: products featured (M:N)
    d["product_ids_featured"] = multi_id_col(PROD_IDS, n, max_ids=8)

    # RELATIONSHIP: segments targeted (M:N)
    d["target_segment_ids"] = multi_id_col(SEG_IDS, n, max_ids=5)

    # RELATIONSHIP: A/B test (M:1)
    d["ab_test_id"] = dirty([f"EXP-{uid()}" if random.random() > 0.4 else "" for _ in range(n)], null_pct=0.2)

    # RELATIONSHIP: triggered by previous email (self-chain)
    d["triggered_by_email_id"] = dirty([f"EM-{uid()}" if random.random() > 0.65 else "" for _ in range(n)], null_pct=0.3)
    d["sequence_position"] = np.random.choice([0,1,2,3,4,5,6,7], n, p=[0.3,0.25,0.15,0.1,0.08,0.05,0.04,0.03])
    d["drip_campaign_id"] = dirty([f"DRIP-{random.randint(1,50)}" if random.random() > 0.5 else "" for _ in range(n)])

    # RELATIONSHIP: conversion outcome
    d["conversion_id"] = dirty([f"CVN-{uid()}" if random.random() > 0.8 else "" for _ in range(n)], null_pct=0.5)

    # Email config (25+ cols)
    d["subject_line_len"] = np.random.randint(5, 100, n)
    d["has_personalization"] = np.random.choice([0, 1, "Y", "N", None], n)
    d["email_type"] = dirty(np.random.choice(["promotional","transactional","newsletter","triggered","welcome","cart_abandon","browse_abandon","reengagement","birthday","anniversary","winback","milestone","flash_sale"], n))
    d["content_category"] = dirty(np.random.choice(["sale","new_arrival","trending","recommendation","loyalty","seasonal","event","education","digest","survey"], n))
    d["n_products"] = np.random.randint(0, 25, n)
    d["n_cta"] = np.random.randint(1, 8, n)
    d["discount_pct"] = np.random.choice([0,5,10,15,20,25,30,40,50,60,70], n)
    d["coupon_code"] = dirty([f"PROMO{random.randint(100,999)}" if random.random() > 0.5 else "" for _ in range(n)])
    d["html_size_kb"] = np.round(np.random.exponential(60, n), 1)

    # Delivery (10+ cols)
    d["delivery_status"] = dirty(np.random.choice(["delivered","bounced_hard","bounced_soft","deferred","dropped","blocked","throttled"], n, p=[0.82,0.04,0.04,0.03,0.03,0.02,0.02]))
    d["spam_score"] = np.round(np.random.uniform(0, 8, n), 2)
    d["inbox_placement"] = dirty(np.random.choice(["primary","promotions","social","spam","updates","focused","other"], n))

    # Engagement (6 × 26 = 156 cols)
    d.update(bulk_metrics(n, ["open","click","read","scroll","fwd","reply"], scale=10))

    # Core engagement (12 cols)
    d["is_opened"] = np.random.choice([0, 1], n, p=[0.72, 0.28])
    d["open_count"] = np.where(np.array(d["is_opened"]) == 1, np.random.poisson(2, n), 0)
    d["is_clicked"] = np.where(np.array(d["is_opened"]) == 1, np.random.choice([0, 1], n, p=[0.65, 0.35]), 0)
    d["clicked_product_id"] = dirty([random.choice(PROD_IDS) if random.random() > 0.6 else "" for _ in range(n)])  # → product!
    d["unsubscribed"] = np.random.choice([0, 1], n, p=[0.97, 0.03])
    d["marked_spam"] = np.random.choice([0, 1], n, p=[0.98, 0.02])
    d["converted"] = np.where(np.array(d["is_clicked"]) == 1, np.random.choice([0, 1], n, p=[0.75, 0.25]), 0)
    d["conv_value_krw"] = np.where(np.array(d["converted"]) == 1, np.round(np.random.exponential(50000, n), 0), 0)
    d["variant"] = np.random.choice(["control","A","B","C","D","E"], n)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "email_marketing_detail.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 5. web_analytics_sessions.csv (~300 cols, 12000 rows)
# RELATIONSHIPS: → customer(M:1), → campaign(via UTM M:1),
#                → products viewed(M:N), → conversion(M:1),
#                → previous_session(self-chain), → impression(M:N)
# ═══════════════════════════════════════════════════════════════
def gen_web():
    print("  [5/15] web_analytics_sessions.csv ...")
    n = 12000
    d = {}

    d["session_id"] = [f"SES-{uid()}" for _ in range(n)]
    d["visitor_id"] = dirty([random.choice(CUST_IDS) if random.random() > 0.35 else f"anon_{uid()}" for _ in range(n)])
    d["ga_client_id"] = [f"GA1.2.{random.randint(1000000,9999999)}.{random.randint(1000000000,1999999999)}" for _ in range(n)]
    d["session_start_iso"] = [pd.Timestamp(np.random.choice(dates)).isoformat() for _ in range(n)]

    # RELATIONSHIP: campaign via UTM
    d["campaign_utm"] = dirty([random.choice(CAMP_IDS) if random.random() > 0.35 else random.choice(["","(not set)","(direct)"]) for _ in range(n)])

    # RELATIONSHIP: products viewed in session (M:N)
    d["products_viewed_ids"] = multi_id_col(PROD_IDS, n, max_ids=12)
    d["products_carted_ids"] = multi_id_col(PROD_IDS, n, max_ids=5)
    d["products_purchased_ids"] = dirty(multi_id_col(PROD_IDS, n, max_ids=3), null_pct=0.5)

    # RELATIONSHIP: conversion (M:1)
    d["conversion_id"] = dirty([f"CVN-{uid()}" if random.random() > 0.82 else "" for _ in range(n)], null_pct=0.3)

    # RELATIONSHIP: previous session (self-chain)
    d["prev_session_id"] = dirty([f"SES-{uid()}" if random.random() > 0.4 else "" for _ in range(n)], null_pct=0.2)
    d["session_sequence_num"] = np.random.poisson(3, n) + 1

    # RELATIONSHIP: impressions that led to this session (M:N)
    d["attributed_impression_ids"] = multi_id_col([f"IMP-{uid()}" for _ in range(200)], n, max_ids=5)

    # RELATIONSHIP: email that drove this session
    d["source_email_id"] = dirty([f"EM-{uid()}" if random.random() > 0.75 else "" for _ in range(n)], null_pct=0.4)

    # RELATIONSHIP: matched segments
    d["visitor_segment_ids"] = multi_id_col(SEG_IDS, n, max_ids=5)

    # Traffic (15+ cols)
    d["source"] = dirty(np.random.choice(["google","naver","daum","facebook","instagram","youtube","direct","email","kakao","tiktok","twitter","linkedin","affiliate","referral","(direct)","(not set)"], n))
    d["medium"] = dirty(np.random.choice(["organic","cpc","cpm","email","social","referral","direct","affiliate","display","video","push","sms","(none)"], n))
    d["channel_grouping"] = dirty(np.random.choice(["Organic Search","Paid Search","Social","Email","Direct","Referral","Display","Video","Affiliate","Other"], n))
    d["landing_page"] = dirty(np.random.choice(["/","/product/123","/category/fashion","/event/sale","/signup","/blog/post-1","/search?q=shoes","/cart","/checkout","/mypage"], n))
    d["gclid"] = dirty([f"gclid_{uid()}" if random.random() > 0.7 else "" for _ in range(n)], null_pct=0.5)
    d["fbclid"] = dirty([f"fbclid_{uid()}" if random.random() > 0.8 else "" for _ in range(n)], null_pct=0.6)

    # Session metrics (10 × 26 = 260 cols)
    d.update(bulk_metrics(n, ["pv","upv","duration","events","scroll","click","form","video","search","cart"], scale=30))

    # Device & perf (15+ cols)
    d["device_cat"] = dirty(np.random.choice(["mobile","desktop","tablet","smart_tv","unknown"], n))
    d["os_name"] = dirty(np.random.choice(OS_LIST + ["(not set)"], n))
    d["browser_name"] = dirty(np.random.choice(BROWSERS + ["(not set)","bot"], n))
    d["country_code"] = dirty(np.random.choice(COUNTRIES + ["(not set)"], n))
    d["region_name"] = dirty(np.random.choice(REGIONS + ["(not set)"], n))
    d["page_load_ms"] = np.random.exponential(2500, n).astype(int)
    d["lcp_ms"] = np.random.exponential(2000, n).astype(int)
    d["cls"] = np.round(np.random.exponential(0.1, n), 3)

    # Ecommerce (15+ cols)
    d["transactions"] = np.random.choice([0,1,2,3], n, p=[0.82,0.14,0.03,0.01])
    d["revenue_krw"] = np.where(np.array(d["transactions"]) > 0, np.round(np.random.exponential(80000, n), 0), 0)
    d["items_purchased"] = np.where(np.array(d["transactions"]) > 0, np.random.poisson(2, n), 0)
    d["cart_value_krw"] = np.round(np.random.exponential(50000, n), 0)
    d["payment_method"] = np.where(np.array(d["transactions"]) > 0, np.random.choice(["credit","kakao_pay","naver_pay","toss","bank","phone","point","paypal","apple_pay"], n), "")

    # User behavior (8 cols)
    d["is_new"] = np.random.choice([0, 1, "new", "returning", None], n)
    d["visit_number"] = np.random.poisson(5, n) + 1
    d["user_segment"] = dirty(np.random.choice(SEGMENTS, n))
    d["engagement_level"] = dirty(np.random.choice(["high","medium","low","churning","new","bot_suspect"], n))

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "web_analytics_sessions.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 6. social_media_performance.csv (~300 cols, 8000 rows)
# RELATIONSHIPS: → campaign(M:N!), → influencer(M:N),
#                → products tagged(M:N), → creative(M:N),
#                → competitor mentions(M:N), → parent_post(self)
# ═══════════════════════════════════════════════════════════════
def gen_social():
    print("  [6/15] social_media_performance.csv ...")
    n = 8000
    d = {}

    d["post_id"] = [f"POST-{uid()}" for _ in range(n)]

    # RELATIONSHIP: campaigns (M:N! — one post can serve multiple campaigns)
    d["brand_campaign_ref"] = dirty(multi_id_col(CAMP_IDS, n, max_ids=3))
    d["primary_campaign"] = dirty([random.choice(CAMP_IDS) if random.random() > 0.25 else "" for _ in range(n)])

    # RELATIONSHIP: influencer (M:N — collabs have multiple influencers)
    d["influencer_ids"] = dirty(multi_id_col(INFLR_IDS, n, max_ids=4))
    d["primary_influencer"] = dirty([random.choice(INFLR_IDS) if random.random() > 0.5 else "" for _ in range(n)])

    # RELATIONSHIP: products tagged (M:N)
    d["tagged_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=10)
    d["shop_product_ids"] = dirty(multi_id_col(PROD_IDS, n, max_ids=5), null_pct=0.3)

    # RELATIONSHIP: creative assets used (M:N)
    d["creative_asset_ids"] = multi_id_col(CREAT_IDS, n, max_ids=5)

    # RELATIONSHIP: parent post (self-reference: thread, repost, reply)
    d["parent_post_id"] = dirty([f"POST-{uid()}" if random.random() > 0.7 else "" for _ in range(n)], null_pct=0.3)
    d["thread_root_id"] = dirty([f"POST-{uid()}" if random.random() > 0.8 else "" for _ in range(n)], null_pct=0.4)
    d["repost_of_id"] = dirty([f"POST-{uid()}" if random.random() > 0.85 else "" for _ in range(n)], null_pct=0.5)

    # RELATIONSHIP: competitor posts referenced
    d["competitor_post_refs"] = dirty(multi_id_col([f"COMP_POST-{uid()}" for _ in range(100)], n, max_ids=3), null_pct=0.5)
    d["competitor_brand_ids"] = dirty(multi_id_col([f"COMP-{i}" for i in range(1,31)], n, max_ids=3), null_pct=0.4)

    # RELATIONSHIP: audience lists targeted (for paid posts)
    d["audience_list_ids"] = dirty(multi_id_col(AUD_IDS, n, max_ids=4), null_pct=0.3)

    d["platform"] = dirty(np.random.choice(["instagram","facebook","youtube","tiktok","twitter_x","linkedin","naver_blog","kakao_story","threads","pinterest","reddit","snapchat"], n))
    d["account_id"] = dirty([f"ACC-{random.randint(1,30)}" for _ in range(n)])
    d["post_date"] = rdates(n, "%Y%m%d")
    d["content_type"] = dirty(np.random.choice(["image","video","carousel","reel","story","live","text","poll","quiz","infographic","thread","collab_post","shop_post","AR_filter"], n))
    d["is_sponsored"] = np.random.choice([0, 1, "sponsored", "organic", None], n)
    d["duration_sec"] = np.where(np.random.random(n) > 0.4, np.random.randint(3, 600, n), 0)

    # Engagement (12 × 17 = 204 cols)
    limited_suffixes = ["_raw","_norm","_pct","_delta","_yoy","_mom","_avg7d","_avg30d",
                        "_median","_p75","_p95","_std","_min","_max","_organic","_paid","_cumul"]
    d.update(bulk_metrics(n, ["impr","reach","view","like","comment","share","save","click","follow","unfollow","reply","repost"], suffixes=limited_suffixes, scale=50))

    # Rates & cost (12 cols)
    d["engagement_rate"] = np.round(np.random.uniform(0.1, 20, n), 3)
    d["virality_rate"] = np.round(np.random.exponential(0.5, n), 4)
    d["ctr"] = np.round(np.random.uniform(0.01, 8, n), 3)
    d["cpe_krw"] = np.round(np.random.exponential(200, n), 0)
    d["ad_spend_krw"] = np.round(np.random.exponential(150000, n), 0)
    d["sentiment_score"] = np.round(np.random.uniform(-1, 1, n), 3)
    d["sentiment_label"] = dirty(np.random.choice(["very_positive","positive","neutral","negative","very_negative","mixed","controversial"], n))

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "social_media_performance.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 7. conversion_funnel.csv (~300 cols, 10000 rows)
# RELATIONSHIPS: → customer(M:1), → campaign(M:1), → product(M:1),
#                → all channels touched(M:N), → session(M:1),
#                → email(M:1), → impression(M:1), → coupon(M:1),
#                → order→return→exchange chain, → referral_customer
# ═══════════════════════════════════════════════════════════════
def gen_conversions():
    print("  [7/15] conversion_funnel.csv ...")
    n = 10000
    d = {}

    d["conversion_id"] = [f"CVN-{uid()}" for _ in range(n)]
    d["client_id"] = dirty([random.choice(CUST_IDS) for _ in range(n)])
    d["marketing_campaign"] = dirty([random.choice(CAMP_IDS) for _ in range(n)])
    d["product_sku"] = dirty([random.choice(PROD_IDS) for _ in range(n)])
    d["order_id"] = [f"ORD-{uid()}" for _ in range(n)]
    d["event_datetime_kst"] = [f"{pd.Timestamp(np.random.choice(dates)).strftime('%Y-%m-%d %H:%M:%S')}+09:00" for _ in range(n)]

    # RELATIONSHIP: all products in this order (M:N)
    d["order_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=8)

    # RELATIONSHIP: all campaigns that touched this customer (M:N)
    d["attributed_campaign_ids"] = multi_id_col(CAMP_IDS, n, max_ids=6)

    # RELATIONSHIP: touchpoints (M:N for each channel)
    for ch in CHANNELS:
        d[f"touch_{ch}"] = np.random.poisson(0.7, n)

    # RELATIONSHIP: specific session, email, impression that drove conversion
    d["last_session_id"] = dirty([f"SES-{uid()}" for _ in range(n)], null_pct=0.1)
    d["last_email_id"] = dirty([f"EM-{uid()}" if random.random() > 0.5 else "" for _ in range(n)], null_pct=0.3)
    d["last_impression_id"] = dirty([f"IMP-{uid()}" if random.random() > 0.4 else "" for _ in range(n)], null_pct=0.2)
    d["coupon_id"] = dirty([f"PROMO{random.randint(100,999)}" if random.random() > 0.5 else "" for _ in range(n)])

    # RELATIONSHIP: referral (which customer referred this buyer?)
    d["referred_by_customer_id"] = dirty([random.choice(CUST_IDS) if random.random() > 0.8 else "" for _ in range(n)], null_pct=0.3)

    # RELATIONSHIP: return/exchange chain (self-reference)
    d["return_order_id"] = dirty([f"ORD-{uid()}" if random.random() > 0.85 else "" for _ in range(n)], null_pct=0.5)
    d["exchange_order_id"] = dirty([f"ORD-{uid()}" if random.random() > 0.9 else "" for _ in range(n)], null_pct=0.6)
    d["original_order_id"] = dirty([f"ORD-{uid()}" if random.random() > 0.88 else "" for _ in range(n)], null_pct=0.5)

    # RELATIONSHIP: loyalty event
    d["loyalty_event_id"] = dirty([f"LYL-{uid()}" if random.random() > 0.6 else "" for _ in range(n)], null_pct=0.3)

    # Funnel (10 cols)
    d["funnel_stage"] = dirty(np.random.choice(["awareness","interest","consideration","intent","evaluation","purchase","post_purchase","advocacy","churn"], n))
    d["entry_point"] = dirty(np.random.choice(["homepage","product","category","search","landing","deeplink","push","email","social","app","qr_code","offline"], n))
    d["first_touch_ch"] = dirty(np.random.choice(CHANNELS, n))
    d["last_touch_ch"] = dirty(np.random.choice(CHANNELS, n))
    d["total_touchpoints"] = np.random.poisson(7, n) + 1
    d["days_in_funnel"] = np.random.exponential(21, n).astype(int)

    # Multi-touch attribution (8 × 3 = 24 cols)
    for model in ["last_click","first_click","linear","time_decay","position","data_driven","markov","shapley"]:
        d[f"attr_{model}_credit"] = np.round(np.random.uniform(0, 1, n), 4)
        d[f"attr_{model}_rev"] = np.round(np.random.exponential(25000, n), 0)
        d[f"attr_{model}_roas"] = np.round(np.random.exponential(2, n), 2)

    # Revenue metrics (6 × 26 = 156 cols)
    d.update(bulk_metrics(n, ["gross_rev","net_rev","margin","cac","ltv","roas"], scale=30000))

    # Post-conversion (12 cols)
    d["delivered"] = np.random.choice([0, 1], n, p=[0.08, 0.92])
    d["returned"] = np.random.choice([0, 1], n, p=[0.86, 0.14])
    d["return_reason"] = dirty(np.where(np.array(d["returned"]) == 1, np.random.choice(["size","quality","wrong","changed_mind","defect","late","color","damaged"], n), ""))
    d["review_rating"] = np.random.choice([0,1,2,3,4,5], n, p=[0.55,0.02,0.03,0.08,0.15,0.17])
    d["nps_score"] = np.random.randint(-1, 11, n)
    d["repurchased_30d"] = np.random.choice([0, 1, None], n)
    d["repurchased_90d"] = np.random.choice([0, 1, None], n)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "conversion_funnel.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 8. crm_interactions.csv (~250 cols, 8000 rows)
# RELATIONSHIPS: → customer(M:1), → campaign(M:1), → agent(M:1),
#                → parent_interaction(self-chain), → order(M:1),
#                → product(M:N), → escalation_path(M:N agents),
#                → related_interactions(M:N)
# ═══════════════════════════════════════════════════════════════
def gen_crm():
    print("  [8/15] crm_interactions.csv ...")
    n = 8000
    d = {}

    d["interaction_id"] = [f"INT-{uid()}" for _ in range(n)]
    d["account_no"] = dirty([random.choice(CUST_IDS) for _ in range(n)])
    d["mkt_initiative_id"] = dirty([random.choice(CAMP_IDS) if random.random() > 0.25 else "" for _ in range(n)])
    d["agent_id"] = dirty([f"AGT-{random.randint(1,80)}" for _ in range(n)])
    d["interaction_date"] = rdates(n, "%y.%m.%d")

    # RELATIONSHIP: parent interaction (self-chain — follow-ups)
    d["parent_interaction_id"] = dirty([f"INT-{uid()}" if random.random() > 0.55 else "" for _ in range(n)], null_pct=0.2)
    d["thread_root_id"] = dirty([f"INT-{uid()}" if random.random() > 0.5 else "" for _ in range(n)], null_pct=0.3)
    d["interaction_depth"] = np.random.choice([0,1,2,3,4,5], n, p=[0.35,0.25,0.2,0.1,0.06,0.04])

    # RELATIONSHIP: related interactions (M:N)
    d["related_interaction_ids"] = multi_id_col([f"INT-{uid()}" for _ in range(200)], n, max_ids=4)

    # RELATIONSHIP: order (the order this interaction is about)
    d["related_order_id"] = dirty([f"ORD-{uid()}" if random.random() > 0.4 else "" for _ in range(n)], null_pct=0.2)

    # RELATIONSHIP: products discussed (M:N)
    d["discussed_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=5)

    # RELATIONSHIP: escalation path (ordered list of agents)
    d["escalation_agent_ids"] = dirty(["|".join([f"AGT-{random.randint(1,80)}" for _ in range(random.randint(0,4))]) if random.random() > 0.6 else "" for _ in range(n)])

    # RELATIONSHIP: team
    d["team_id"] = dirty([random.choice(TEAM_IDS) for _ in range(n)])

    # Interaction details (25+ cols)
    d["interaction_type"] = dirty(np.random.choice(["inbound_call","outbound_call","email","chat","social_dm","in_store","video","callback","sms","push","chatbot","voicebot","ticket"], n))
    d["channel"] = dirty(np.random.choice(["phone","email","live_chat","kakao","line","fb_messenger","ig_dm","in_person","video","chatbot","voicebot","zendesk"], n))
    d["category"] = dirty(np.random.choice(["inquiry","complaint","support","sales","retention","upsell","feedback","billing","shipping","return","exchange","technical","account","loyalty","warranty","fraud","privacy"], n))
    d["priority"] = dirty(np.random.choice(["critical","high","medium","low","P0","P1","P2","P3","긴급","보통","낮음"], n))
    d["sentiment"] = dirty(np.random.choice(["very_positive","positive","neutral","negative","very_negative","angry","frustrated","confused","satisfied","delighted"], n))
    d["resolution"] = dirty(np.random.choice(["resolved_first","resolved_followup","escalated","pending","transferred","abandoned","auto_resolved","partial"], n))
    d["resolution_time_min"] = np.random.exponential(45, n).astype(int)
    d["handle_time_min"] = np.random.exponential(20, n).astype(int)
    d["csat_score"] = np.random.choice([0,1,2,3,4,5,-1,None], n)
    d["nps_score"] = np.random.choice(list(range(0,11)) + [-1, None], n)

    # Context scores (5 × 5 = 25 cols)
    d.update(bulk_scores(n, ["churn_risk","ltv","engagement","effort","satisfaction"]))

    # Behavioral (6 × 26 = 156 cols)
    d.update(bulk_metrics(n, ["ticket","call","email_crm","chat","escalation","resolution"], scale=5))

    # Outcome (8 cols)
    d["outcome"] = dirty(np.random.choice(["sale","retention","upsell","cross_sell","info","resolved","escalated","no_resolution","cancelled","churn_prevented","refund_issued"], n))
    d["revenue_impact_krw"] = np.round(np.random.normal(0, 60000, n), 0)
    d["coupon_issued"] = np.random.choice([0, 1, None], n)
    d["next_best_action"] = dirty(np.random.choice(["upsell","cross_sell","loyalty_upgrade","retention","survey","no_action","education","referral","escalate"], n))

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "crm_interactions.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 9. product_catalog.csv (~250 cols, 3000 rows)
# RELATIONSHIPS: → bundle_products(M:N self), → category hierarchy,
#                → supplier(M:1), → campaigns featured in(M:N),
#                → substitute_products(M:N), → complementary_products(M:N)
# ═══════════════════════════════════════════════════════════════
def gen_products():
    print("  [9/15] product_catalog.csv ...")
    n = 3000
    d = {}

    d["item_code"] = PROD_IDS + [f"PRD-{i}" for i in range(N_PROD+1, n+1)]

    # RELATIONSHIP: bundle (M:N self-reference)
    d["bundle_product_ids"] = ["|".join(PROD_BUNDLE.get(f"PRD-{i}", [])) if f"PRD-{i}" in PROD_BUNDLE else "" for i in range(1, n+1)]
    d["is_bundle"] = [1 if len(PROD_BUNDLE.get(f"PRD-{i}", [])) > 0 else 0 for i in range(1, n+1)]
    d["parent_product_id"] = dirty([random.choice(PROD_IDS) if random.random() > 0.8 else "" for _ in range(n)], null_pct=0.4)  # variant parent

    # RELATIONSHIP: substitute products (M:N)
    d["substitute_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=5)

    # RELATIONSHIP: complementary products (M:N — "frequently bought together")
    d["complementary_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=8)

    # RELATIONSHIP: campaigns featured in (M:N)
    d["featured_in_campaign_ids"] = multi_id_col(CAMP_IDS, n, max_ids=10)

    # RELATIONSHIP: supplier
    d["supplier_id"] = dirty([f"SUP-{random.randint(1,100)}" for _ in range(n)])
    d["manufacturer_id"] = dirty([f"MFG-{random.randint(1,50)}" for _ in range(n)])

    # RELATIONSHIP: category hierarchy (multiple levels referencing each other)
    d["category_l1_id"] = [f"CAT1-{random.randint(1,15)}" for _ in range(n)]
    d["category_l2_id"] = [f"CAT2-{random.randint(1,60)}" for _ in range(n)]
    d["category_l3_id"] = [f"CAT3-{random.randint(1,200)}" for _ in range(n)]
    d["category_l4_id"] = [f"CAT4-{random.randint(1,500)}" for _ in range(n)]

    d["sku"] = dirty([f"SKU-{random.randint(100000,999999)}" for _ in range(n)])
    d["barcode"] = dirty([f"{random.randint(8800000000000,8899999999999)}" for _ in range(n)])
    d["product_name"] = dirty([f"상품_{random.choice(['프리미엄','베이직','클래식','모던','럭셔리','에센셜','프로','울트라'])}_{i}" for i in range(1, n+1)])
    d["brand"] = dirty(np.random.choice(["브랜드A","브랜드B","브랜드C","브랜드D","PB","콜라보","Brand_X","Nike","Adidas","Samsung","Apple","ZARA","H&M"], n))

    # Pricing (silo: USD!)
    d["list_price_usd"] = np.round(np.random.exponential(50, n), 2)
    d["sale_price_usd"] = np.round(np.array(d["list_price_usd"]) * np.random.uniform(0.4, 1.0, n), 2)
    d["cost_price_usd"] = np.round(np.array(d["list_price_usd"]) * np.random.uniform(0.15, 0.5, n), 2)
    d["margin_pct"] = np.round((1 - np.array(d["cost_price_usd"]) / np.maximum(np.array(d["sale_price_usd"]), 0.01)) * 100, 2)
    d["member_price_usd"] = np.round(np.array(d["sale_price_usd"]) * 0.9, 2)
    d["vip_price_usd"] = np.round(np.array(d["sale_price_usd"]) * 0.85, 2)

    # Inventory (8 cols)
    d["stock_total"] = np.random.poisson(150, n)
    d["stock_available"] = np.maximum(np.array(d["stock_total"]) - np.random.poisson(40, n), 0)
    d["warehouse"] = dirty(np.random.choice(["WH_서울","WH_부산","WH_인천","WH_대전","WH_해외","WH_US","WH_JP","3PL"], n))
    d["is_active"] = np.random.choice([0, 1, "Y", "N", "true", "false", None], n)
    d["launch_date"] = mixed_dates(n)

    # Performance (8 × 26 = 208 cols)
    d.update(bulk_metrics(n, ["units_sold","revenue","views","cart_adds","conv_rate","return_rate","wishlist","review_count"], scale=50))

    # Reviews (6 cols)
    d["avg_rating"] = np.round(np.random.uniform(1.5, 5.0, n), 1)
    d["total_reviews"] = np.random.poisson(30, n)
    d["photo_reviews"] = np.random.poisson(8, n)
    d["sentiment_pos_pct"] = np.round(np.random.uniform(30, 98, n), 1)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "product_catalog.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# 10-15: Remaining files (budget, influencer, ab_test, competitor, content, loyalty)
# Each with complex inter-file relationships
# ═══════════════════════════════════════════════════════════════

def gen_budget():
    print("  [10/15] budget_performance.csv ...")
    n = 5000
    d = {}
    d["budget_line_id"] = [f"BL-{uid()}" for _ in range(n)]
    d["initiative_ref"] = dirty([random.choice(CAMP_IDS) for _ in range(n)])
    # RELATIONSHIP: deal (M:1)
    d["deal_id"] = dirty([random.choice(DEAL_IDS) if random.random() > 0.5 else "" for _ in range(n)])
    # RELATIONSHIP: agency (M:1)
    d["agency_id"] = dirty([random.choice(AGENCY_IDS) for _ in range(n)])
    # RELATIONSHIP: team (M:1)
    d["team_id"] = dirty([random.choice(TEAM_IDS) for _ in range(n)])
    # RELATIONSHIP: product line (M:N)
    d["product_line_ids"] = multi_id_col(PROD_IDS, n, max_ids=5)
    # RELATIONSHIP: influencer costs
    d["influencer_cost_ref"] = dirty([random.choice(INFLR_IDS) if random.random() > 0.7 else "" for _ in range(n)])

    d["fiscal_year"] = np.random.choice(["FY2023","FY2024","FY2025","2023","2024","2025","23기","24기","25기"], n)
    d["fiscal_quarter"] = np.random.choice(["Q1","Q2","Q3","Q4","1분기","2분기","3분기","4분기"], n)
    d["report_date"] = mixed_dates(n)
    d["department"] = dirty(np.random.choice(["퍼포먼스마케팅","브랜드마케팅","CRM","그로스","콘텐츠","데이터","크리에이티브","미디어","PR","이벤트"], n))

    # Budget (17 cat × 3 = 51 cols) — JPY!
    for cat in ["media","creative","tech","personnel","agency","production","research",
                "events","sponsorship","influencer","content","tools","data","testing",
                "training","travel","misc"]:
        d[f"plan_{cat}_jpy"] = np.round(np.random.exponential(800000, n), 0)
        d[f"actual_{cat}_jpy"] = np.round(np.array(d[f"plan_{cat}_jpy"]) * np.random.uniform(0.3, 1.8, n), 0)
        d[f"var_{cat}_pct"] = np.round((np.array(d[f"actual_{cat}_jpy"]) / np.maximum(np.array(d[f"plan_{cat}_jpy"]), 1) - 1) * 100, 2)

    # Channel spend (20 × 2 = 40 cols)
    for ch in CHANNELS:
        d[f"spend_{ch}_jpy"] = np.round(np.random.exponential(150000, n), 0)
        d[f"spend_{ch}_pct"] = np.round(np.random.uniform(0, 0.25, n), 4)

    # KPI (20 × 4 = 80 cols)
    for kpi in ["impressions","clicks","conversions","leads","signups","installs","video_views",
                "engagement","reach","frequency","ctr","cvr","cpc","cpa","cpm","roas","roi",
                "ltv_cac","payback_mo","brand_lift"]:
        d[f"kpi_{kpi}_tgt"] = np.round(np.random.exponential(1500, n), 2)
        d[f"kpi_{kpi}_act"] = np.round(np.random.exponential(1500, n), 2)
        d[f"kpi_{kpi}_ach"] = np.round(np.random.uniform(0.1, 3, n), 3)
        d[f"kpi_{kpi}_trend"] = dirty(np.random.choice(["improving","declining","stable","volatile","개선","하락"], n))

    d["forecast_accuracy"] = np.round(np.random.uniform(40, 100, n), 1)
    d["efficiency_rating"] = dirty(np.random.choice(["A","B","C","D","F","S","A+"], n))

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "budget_performance.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


def gen_influencers():
    print("  [11/15] influencer_partnerships.csv ...")
    n = 3000
    d = {}
    d["partnership_id"] = [f"PRTN-{uid()}" for _ in range(n)]
    d["influencer_ref"] = dirty([random.choice(INFLR_IDS) for _ in range(n)])
    # RELATIONSHIP: campaign (M:N — one partnership can span campaigns)
    d["campaign_link_ids"] = multi_id_col(CAMP_IDS, n, max_ids=5)
    d["primary_campaign"] = dirty([random.choice(CAMP_IDS) for _ in range(n)])
    # RELATIONSHIP: products (M:N)
    d["product_ids"] = multi_id_col(PROD_IDS, n, max_ids=10)
    # RELATIONSHIP: agency (M:1)
    d["managing_agency_id"] = dirty([random.choice(AGENCY_IDS) for _ in range(n)])
    # RELATIONSHIP: creative assets produced (M:N)
    d["produced_asset_ids"] = multi_id_col(CREAT_IDS, n, max_ids=8)
    # RELATIONSHIP: target audience (M:N)
    d["target_audience_ids"] = multi_id_col(AUD_IDS, n, max_ids=4)
    # RELATIONSHIP: other influencer collabs (M:N)
    d["collab_influencer_ids"] = dirty(multi_id_col(INFLR_IDS, n, max_ids=3), null_pct=0.4)
    # RELATIONSHIP: competitor counter (which competitor influencers)
    d["competitor_influencer_ids"] = dirty(multi_id_col([f"COMP_INF-{i}" for i in range(1,50)], n, max_ids=3), null_pct=0.5)

    d["contract_date"] = mixed_dates(n)
    d["platform_primary"] = dirty(np.random.choice(["instagram","youtube","tiktok","naver_blog","twitter_x","twitch","podcast","newsletter"], n))
    d["followers_total"] = np.random.exponential(50000, n).astype(int)
    d["tier"] = dirty(np.random.choice(["mega","macro","mid","micro","nano"], n))
    d["engagement_rate_avg"] = np.round(np.random.uniform(0.5, 15, n), 2)
    d["fake_follower_pct"] = np.round(np.random.exponential(5, n), 1)
    d["fee_krw"] = np.round(np.random.exponential(500000, n), 0)
    d["fee_usd"] = np.round(np.array(d["fee_krw"]) / np.random.choice([1100,1200,1300,1400], n), 2)  # inconsistent exchange rates!
    d["payment_status"] = dirty(np.random.choice(["paid","pending","invoiced","overdue","cancelled","disputed"], n))

    # Performance (8 × 26 = 208 cols)
    d.update(bulk_metrics(n, ["post_reach","post_engage","post_click","post_conv","story_view","video_view","comment_inf","save_inf"], scale=500))

    d["roi"] = np.round(np.random.uniform(-50, 500, n), 2)
    d["emv_krw"] = np.round(np.random.exponential(1000000, n), 0)
    d["brand_lift_pct"] = np.round(np.random.uniform(-5, 30, n), 2)
    d["controversy_flag"] = np.random.choice([0, 1, None], n, p=[0.88, 0.07, 0.05])

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "influencer_partnerships.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


def gen_abtests():
    print("  [12/15] ab_test_experiments.csv ...")
    n = 5000
    d = {}
    d["experiment_id"] = [f"EXP-{uid()}" for _ in range(n)]
    # RELATIONSHIP: campaign (M:N — experiment can test across campaigns)
    d["campaign_ids"] = multi_id_col(CAMP_IDS, n, max_ids=4)
    # RELATIONSHIP: creative variants tested (M:N)
    d["creative_variant_ids"] = multi_id_col(CREAT_IDS, n, max_ids=6)
    # RELATIONSHIP: audience lists (M:N)
    d["test_audience_ids"] = multi_id_col(AUD_IDS, n, max_ids=4)
    # RELATIONSHIP: product (tested product pages)
    d["tested_product_ids"] = dirty(multi_id_col(PROD_IDS, n, max_ids=5), null_pct=0.3)
    # RELATIONSHIP: parent experiment (iteration/follow-up)
    d["parent_experiment_id"] = dirty([f"EXP-{uid()}" if random.random() > 0.7 else "" for _ in range(n)], null_pct=0.3)
    # RELATIONSHIP: segments tested (M:N)
    d["segment_ids_tested"] = multi_id_col(SEG_IDS, n, max_ids=5)

    d["test_name"] = dirty([f"테스트_{random.choice(['CTA','색상','레이아웃','헤드라인','이미지','가격','메시지','타이밍'])}_{i}" for i in range(1,n+1)])
    d["test_type"] = dirty(np.random.choice(["A/B","A/B/C","multivariate","bandit","sequential","factorial"], n))
    d["start_date"] = mixed_dates(n)
    d["end_date"] = mixed_dates(n)
    d["status"] = dirty(np.random.choice(["running","completed","paused","cancelled","draft","analyzing","winner_declared","inconclusive"], n))
    d["confidence_level"] = np.random.choice([0.90, 0.95, 0.99], n)

    # Variant results (6 × 12 = 72 cols)
    for v in ["control","variant_a","variant_b","variant_c","variant_d","variant_e"]:
        d[f"{v}_impressions"] = np.random.exponential(5000, n).astype(int)
        d[f"{v}_clicks"] = np.random.exponential(200, n).astype(int)
        d[f"{v}_conversions"] = np.random.exponential(20, n).astype(int)
        d[f"{v}_revenue_krw"] = np.round(np.random.exponential(500000, n), 0)
        d[f"{v}_ctr"] = np.round(np.random.uniform(0.5, 10, n), 3)
        d[f"{v}_cvr"] = np.round(np.random.uniform(0.1, 8, n), 3)
        d[f"{v}_bounce_rate"] = np.round(np.random.uniform(10, 80, n), 2)
        d[f"{v}_engagement"] = np.round(np.random.uniform(1, 20, n), 2)
        d[f"{v}_retention"] = np.round(np.random.uniform(10, 60, n), 2)
        d[f"{v}_ltv_impact"] = np.round(np.random.normal(0, 10000, n), 0)
        d[f"{v}_sample_n"] = np.random.exponential(2000, n).astype(int)
        d[f"{v}_p_value"] = np.round(np.random.exponential(0.1, n), 4)

    # Segment breakdowns (5 × 5 = 25 cols)
    for seg in ["new_user","returning","mobile","desktop","high_value"]:
        d[f"seg_{seg}_cvr_control"] = np.round(np.random.uniform(0.5, 8, n), 3)
        d[f"seg_{seg}_cvr_winner"] = np.round(np.random.uniform(0.5, 10, n), 3)
        d[f"seg_{seg}_lift"] = np.round(np.random.normal(5, 15, n), 2)
        d[f"seg_{seg}_p_value"] = np.round(np.random.exponential(0.15, n), 4)
        d[f"seg_{seg}_n"] = np.random.exponential(1000, n).astype(int)

    d["winner"] = dirty(np.random.choice(["control","variant_a","variant_b","none","inconclusive","TBD"], n))
    d["p_value"] = np.round(np.random.exponential(0.1, n), 4)
    d["effect_size"] = np.round(np.random.normal(0, 0.5, n), 4)
    d["lift_pct"] = np.round(np.random.normal(5, 20, n), 2)
    d["deployed"] = np.random.choice([0, 1, "Y", "N", "partial", None], n)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "ab_test_experiments.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


def gen_competitors():
    print("  [13/15] competitor_intelligence.csv ...")
    n = 3000
    d = {}
    d["report_id"] = [f"CI-{uid()}" for _ in range(n)]
    d["competitor_id"] = dirty([f"COMP-{random.randint(1,30)}" for _ in range(n)])
    d["our_campaign_ref"] = dirty([random.choice(CAMP_IDS) if random.random() > 0.4 else "" for _ in range(n)])
    # RELATIONSHIP: our products compared (M:N)
    d["our_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=5)
    # RELATIONSHIP: competitor products
    d["competitor_product_ids"] = multi_id_col([f"COMP_PRD-{i}" for i in range(1,200)], n, max_ids=5)
    # RELATIONSHIP: channels analyzed (M:N)
    d["channels_analyzed"] = multi_id_col(CHANNELS, n, max_ids=6)
    # RELATIONSHIP: our influencers they poached
    d["poached_influencer_ids"] = dirty(multi_id_col(INFLR_IDS, n, max_ids=3), null_pct=0.6)

    d["report_date"] = mixed_dates(n)
    d["source"] = dirty(np.random.choice(["semrush","similarweb","socialbakers","manual","scraping","news","industry_report","ad_library"], n))

    # SOV (20 × 2 = 40 cols)
    for ch in CHANNELS:
        d[f"sov_{ch}_pct"] = np.round(np.random.uniform(0, 30, n), 2)
        d[f"sov_{ch}_trend"] = dirty(np.random.choice(["up","down","stable","volatile","급증","급감"], n))

    # Metrics (5 × 24 = 120 cols)
    comp_suffixes = ["_raw","_norm","_pct","_delta","_yoy","_mom","_avg7d","_avg30d",
                     "_median","_p75","_std","_min","_max","_organic","_paid","_cumul",
                     "_our_val","_comp_val","_gap","_rank","_index","_score","_trend_val","_benchmark"]
    d.update(bulk_metrics(n, ["traffic","social","search","brand","pricing"], suffixes=comp_suffixes, scale=100))

    d["threat_level"] = dirty(np.random.choice(["low","medium","high","critical","emerging","declining"], n))
    d["market_share_pct"] = np.round(np.random.uniform(1, 35, n), 2)
    d["price_index"] = np.round(np.random.uniform(70, 150, n), 1)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "competitor_intelligence.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


def gen_content():
    print("  [14/15] content_asset_library.csv ...")
    n = 5000
    d = {}
    d["asset_id"] = [f"AST-{uid()}" for _ in range(n)]
    d["creative_ref"] = dirty([random.choice(CREAT_IDS) for _ in range(n)])
    # RELATIONSHIP: campaigns (M:N)
    d["linked_campaign_ids"] = multi_id_col(CAMP_IDS, n, max_ids=8)
    # RELATIONSHIP: products (M:N)
    d["linked_product_ids"] = multi_id_col(PROD_IDS, n, max_ids=6)
    # RELATIONSHIP: influencer who created (M:1)
    d["created_by_influencer"] = dirty([random.choice(INFLR_IDS) if random.random() > 0.6 else "" for _ in range(n)])
    # RELATIONSHIP: parent asset (version history self-chain)
    d["parent_asset_id"] = dirty([f"AST-{uid()}" if random.random() > 0.5 else "" for _ in range(n)], null_pct=0.2)
    d["derived_from_asset_ids"] = dirty(multi_id_col([f"AST-{uid()}" for _ in range(200)], n, max_ids=3), null_pct=0.4)
    # RELATIONSHIP: A/B test using this asset
    d["used_in_experiment_ids"] = dirty(multi_id_col([f"EXP-{uid()}" for _ in range(200)], n, max_ids=3), null_pct=0.4)
    # RELATIONSHIP: audience targeted with this asset
    d["target_audience_ids"] = dirty(multi_id_col(AUD_IDS, n, max_ids=4), null_pct=0.3)
    # RELATIONSHIP: agency that produced
    d["producing_agency_id"] = dirty([random.choice(AGENCY_IDS) for _ in range(n)])

    d["created_date"] = mixed_dates(n)
    d["asset_type"] = dirty(np.random.choice(["image","video","gif","text","audio","html5","3d_model","ar_filter","template","infographic","animation","interactive"], n))
    d["format"] = dirty(np.random.choice(["jpg","png","webp","mp4","mov","gif","psd","ai","svg","pdf","html","json"], n))
    d["file_size_kb"] = np.random.exponential(500, n).astype(int)
    d["duration_sec"] = np.where(np.random.random(n) > 0.5, np.random.randint(3, 300, n), 0)
    d["compliance_status"] = dirty(np.random.choice(["approved","pending","rejected","revision_needed","expired","auto_approved","needs_legal"], n))
    d["version"] = np.random.choice(["v1","v2","v3","v1.1","v2.0","final","final_v2","최종","최종_수정","최종_최종"], n)

    # Performance (6 × 26 = 156 cols)
    d.update(bulk_metrics(n, ["impr_ast","click_ast","engage_ast","conv_ast","cost_ast","quality_ast"], scale=100))

    d["ai_quality_score"] = np.round(np.random.uniform(0, 100, n), 1)
    d["ai_predicted_ctr"] = np.round(np.random.uniform(0, 10, n), 3)
    d["ai_brand_consistency"] = np.round(np.random.uniform(0, 100, n), 1)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "content_asset_library.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


def gen_loyalty():
    print("  [15/15] loyalty_program.csv ...")
    n = 8000
    d = {}
    d["loyalty_event_id"] = [f"LYL-{uid()}" for _ in range(n)]
    d["member_id"] = dirty([random.choice(CUST_IDS) for _ in range(n)])
    # RELATIONSHIP: order (M:1)
    d["related_order_id"] = dirty([f"ORD-{uid()}" if random.random() > 0.3 else "" for _ in range(n)])
    # RELATIONSHIP: campaign (M:1)
    d["related_campaign_id"] = dirty([random.choice(CAMP_IDS) if random.random() > 0.4 else "" for _ in range(n)])
    # RELATIONSHIP: product purchased (M:N)
    d["product_ids"] = multi_id_col(PROD_IDS, n, max_ids=5)
    # RELATIONSHIP: referred by (circular — member referral chain)
    d["referred_by_member_id"] = dirty([random.choice(CUST_IDS) if random.random() > 0.7 else "" for _ in range(n)], null_pct=0.3)
    d["referral_chain_ids"] = dirty(multi_id_col(CUST_IDS, n, max_ids=5), null_pct=0.4)
    # RELATIONSHIP: partner (M:1)
    d["partner_id"] = dirty([f"PARTNER-{random.randint(1,30)}" if random.random() > 0.5 else "" for _ in range(n)])
    # RELATIONSHIP: CRM interaction that triggered
    d["triggered_by_interaction_id"] = dirty([f"INT-{uid()}" if random.random() > 0.7 else "" for _ in range(n)], null_pct=0.4)

    d["event_date"] = rdates(n, "%Y-%m-%d")
    d["event_type"] = dirty(np.random.choice(["earn","redeem","expire","adjust","transfer","bonus","promotion","signup","upgrade","downgrade","cancellation","reinstatement","review_reward","referral_reward","birthday","quest_complete","gamification"], n))
    d["points_earned"] = np.random.exponential(100, n).astype(int)
    d["points_redeemed"] = np.random.exponential(50, n).astype(int)
    d["points_balance"] = np.random.exponential(500, n).astype(int)
    d["tier_at_event"] = dirty(np.random.choice(["bronze","silver","gold","platinum","diamond","vvip","Bronze","Silver","Gold","브론즈","실버","골드"], n))
    d["tier_progress_pct"] = np.round(np.random.uniform(0, 100, n), 1)
    d["earn_channel"] = dirty(np.random.choice(CHANNELS + ["offline","app","website","partner","referral","review","event","game"], n))

    # Scores (10 × 5 = 50 cols)
    d.update(bulk_scores(n, ["engagement","churn_risk","reactivation","lifetime_value",
                              "point_sensitivity","earn_velocity","redeem_velocity","tier_loyalty",
                              "satisfaction","advocacy"]))

    # Benefits (6 × 26 = 156 cols)
    d.update(bulk_metrics(n, ["benefit","coupon_lyl","cashback","discount_lyl","freeship","exclusive"], scale=20))

    # Gamification (8 cols)
    d["badges_earned"] = np.random.poisson(3, n)
    d["streak_days"] = np.random.poisson(5, n)
    d["level"] = np.random.randint(1, 50, n)
    d["xp_total"] = np.random.exponential(5000, n).astype(int)
    d["leaderboard_rank"] = np.random.randint(1, N_CUST, n)
    d["missions_completed"] = np.random.poisson(10, n)

    df = pd.DataFrame(d)
    df.to_csv(os.path.join(OUT_DIR, "loyalty_program.csv"), index=False)
    print(f"    → {df.shape[0]:,} rows, {df.shape[1]} cols")


# ═══════════════════════════════════════════════════════════════
# Post-generation: inject cross-file chaos
# ═══════════════════════════════════════════════════════════════
def inject_chaos():
    print("\n  [CHAOS] Injecting cross-file inconsistencies ...")

    # 1. Duplicate rows with contradictory data in conversion_funnel
    cf = pd.read_csv(os.path.join(OUT_DIR, "conversion_funnel.csv"))
    dupes = cf.sample(80, random_state=42).copy()
    for col in ["funnel_stage","last_touch_ch","review_rating"]:
        if col in dupes.columns:
            dupes[col] = np.random.choice(dupes[col].dropna().unique(), len(dupes)) if len(dupes[col].dropna()) > 0 else dupes[col]
    cf = pd.concat([cf, dupes], ignore_index=True)
    cf.to_csv(os.path.join(OUT_DIR, "conversion_funnel.csv"), index=False)
    print(f"    → conversion_funnel: +{len(dupes)} contradictory duplicates")

    # 2. Orphan customer IDs in CRM
    crm = pd.read_csv(os.path.join(OUT_DIR, "crm_interactions.csv"))
    orphan_mask = np.random.random(len(crm)) < 0.05
    crm.loc[orphan_mask, "account_no"] = [f"C{str(random.randint(90000,99999)).zfill(6)}" for _ in range(orphan_mask.sum())]
    crm.to_csv(os.path.join(OUT_DIR, "crm_interactions.csv"), index=False)
    print(f"    → crm_interactions: {orphan_mask.sum()} orphan customer IDs")

    # 3. Contradictory campaign status/dates
    camp = pd.read_csv(os.path.join(OUT_DIR, "campaign_registry.csv"))
    future_mask = np.random.random(len(camp)) < 0.08
    camp.loc[future_mask, "status"] = "completed"
    camp.loc[future_mask, "end_date"] = pd.Timestamp("2026-06-15").strftime("%d-%m-%Y")
    camp.to_csv(os.path.join(OUT_DIR, "campaign_registry.csv"), index=False)
    print(f"    → campaign_registry: {future_mask.sum()} contradictory status/date")

    # 4. Orphan campaign IDs in budget
    budget = pd.read_csv(os.path.join(OUT_DIR, "budget_performance.csv"))
    orphan_mask = np.random.random(len(budget)) < 0.03
    budget.loc[orphan_mask, "initiative_ref"] = [f"CAMP-{str(random.randint(9000,9999)).zfill(4)}" for _ in range(orphan_mask.sum())]
    budget.to_csv(os.path.join(OUT_DIR, "budget_performance.csv"), index=False)
    print(f"    → budget_performance: {orphan_mask.sum()} orphan campaign IDs")

    print("    → Done\n")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Clean previous
    if os.path.exists(OUT_DIR):
        for f in os.listdir(OUT_DIR):
            if f.endswith(".csv"):
                os.remove(os.path.join(OUT_DIR, f))

    print(f"\n{'='*70}")
    print(f"  Marketing Silo V2 — EXTREME Complexity Dataset Generator")
    print(f"  Output: {OUT_DIR}")
    print(f"{'='*70}\n")

    gen_customers()
    gen_campaigns()
    gen_impressions()
    gen_email()
    gen_web()
    gen_social()
    gen_conversions()
    gen_crm()
    gen_products()
    gen_budget()
    gen_influencers()
    gen_abtests()
    gen_competitors()
    gen_content()
    gen_loyalty()

    inject_chaos()

    # ─── Summary ───
    print(f"{'='*70}")
    print("  DATASET SUMMARY")
    print(f"{'='*70}")
    total_rows = 0
    total_cols = 0
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(OUT_DIR, f), nrows=0)
            rows = sum(1 for _ in open(os.path.join(OUT_DIR, f))) - 1
            print(f"  {f:40s} {rows:>8,} rows × {len(df.columns):>4} cols")
            total_rows += rows
            total_cols += len(df.columns)
    print(f"  {'─'*58}")
    print(f"  {'TOTAL':40s} {total_rows:>8,} rows × {total_cols:>4} cols")

    print(f"\n{'='*70}")
    print("  RELATIONSHIP MAP (Cross-file connections)")
    print(f"{'='*70}")
    print("""
  customer_master ←→ conversion_funnel     (cust_id ↔ client_id)
  customer_master ←→ email_marketing       (cust_id ↔ subscriber_id)
  customer_master ←→ web_analytics         (cust_id ↔ visitor_id)
  customer_master ←→ crm_interactions      (cust_id ↔ account_no)
  customer_master ←→ loyalty_program       (cust_id ↔ member_id)
  customer_master ←→ ad_impressions        (cust_id ↔ user_hash)
  customer_master → customer_master        (referred_by_cust_id, CIRCULAR)
  customer_master → customer_master        (company_contact_cust_id)
  customer_master → product_catalog        (preferred_product_ids, M:N)
  customer_master → segments               (segment_ids, M:N)

  campaign_registry → campaign_registry    (parent_campaign_id, HIERARCHY)
  campaign_registry ←→ ad_impressions      (camp_id ↔ campaign_code)
  campaign_registry ←→ email_marketing     (camp_id ↔ promo_id)
  campaign_registry ←→ web_analytics       (camp_id ↔ campaign_utm)
  campaign_registry ←→ social_media        (camp_id ↔ brand_campaign_ref, M:N)
  campaign_registry ←→ conversion_funnel   (camp_id ↔ marketing_campaign)
  campaign_registry ←→ crm_interactions    (camp_id ↔ mkt_initiative_id)
  campaign_registry ←→ budget_performance  (camp_id ↔ initiative_ref)
  campaign_registry ←→ influencer_parts    (camp_id ↔ campaign_link_ids, M:N)
  campaign_registry ←→ ab_test_experiments (camp_id ↔ campaign_ids, M:N)
  campaign_registry ←→ competitor_intel    (camp_id ↔ our_campaign_ref)
  campaign_registry ←→ content_assets      (camp_id ↔ linked_campaign_ids, M:N)
  campaign_registry ←→ loyalty_program     (camp_id ↔ related_campaign_id)
  campaign_registry → segments             (target_segment_ids, M:N)
  campaign_registry → audiences            (audience_list_ids, M:N)
  campaign_registry → influencers          (influencer_ids, M:N)
  campaign_registry → products             (featured_product_ids, M:N)
  campaign_registry → agencies             (agency_ids, M:N)
  campaign_registry → deals                (media_deal_ids, M:N)
  campaign_registry → creatives            (creative_asset_ids, M:N)
  campaign_registry → campaigns            (related/competing_campaign_ids, M:N)

  ad_impressions → web_analytics           (session_ref → session_id)
  ad_impressions → conversion_funnel       (conversion_ref → conversion_id)
  ad_impressions → products                (product_ids_shown, M:N)
  ad_impressions → audiences               (matched_audience_ids, M:N)
  ad_impressions → segments                (matched_segment_ids, M:N)

  email_marketing → products               (product_ids_featured, M:N)
  email_marketing → email_marketing        (triggered_by_email_id, CHAIN)
  email_marketing → conversion_funnel      (conversion_id)
  email_marketing → audiences              (email_list_id → AUD)
  email_marketing → creatives              (creative_asset_id)

  web_analytics → products                 (products_viewed/carted/purchased, M:N)
  web_analytics → conversion_funnel        (conversion_id)
  web_analytics → web_analytics            (prev_session_id, CHAIN)
  web_analytics → email_marketing          (source_email_id)
  web_analytics → ad_impressions           (attributed_impression_ids, M:N)
  web_analytics → segments                 (visitor_segment_ids, M:N)

  social_media → influencers               (influencer_ids, M:N)
  social_media → products                  (tagged_product_ids, M:N)
  social_media → creatives                 (creative_asset_ids, M:N)
  social_media → social_media              (parent_post/thread/repost, CHAIN)
  social_media → competitors               (competitor_brand_ids, M:N)
  social_media → audiences                 (audience_list_ids, M:N)

  conversion_funnel → products             (order_product_ids, M:N)
  conversion_funnel → campaigns            (attributed_campaign_ids, M:N)
  conversion_funnel → sessions             (last_session_id)
  conversion_funnel → emails               (last_email_id)
  conversion_funnel → impressions          (last_impression_id)
  conversion_funnel → conversion_funnel    (return/exchange/original_order, CHAIN)
  conversion_funnel → customers            (referred_by_customer_id)
  conversion_funnel → loyalty              (loyalty_event_id)

  crm_interactions → products              (discussed_product_ids, M:N)
  crm_interactions → crm_interactions      (parent/thread/related, CHAIN+M:N)
  crm_interactions → orders                (related_order_id)
  crm_interactions → agents                (escalation_agent_ids, M:N ORDERED)

  product_catalog → product_catalog        (bundle/parent/substitute/complementary, M:N SELF)
  product_catalog → campaigns              (featured_in_campaign_ids, M:N)
  product_catalog → suppliers              (supplier_id)
  product_catalog → categories             (l1/l2/l3/l4, HIERARCHY)

  budget → campaigns                       (initiative_ref)
  budget → deals                           (deal_id)
  budget → agencies                        (agency_id)
  budget → products                        (product_line_ids, M:N)
  budget → influencers                     (influencer_cost_ref)

  influencer → campaigns                   (campaign_link_ids, M:N)
  influencer → products                    (product_ids, M:N)
  influencer → agencies                    (managing_agency_id)
  influencer → creatives                   (produced_asset_ids, M:N)
  influencer → influencers                 (collab_influencer_ids, M:N SELF)
  influencer → audiences                   (target_audience_ids, M:N)

  ab_test → campaigns                      (campaign_ids, M:N)
  ab_test → creatives                      (creative_variant_ids, M:N)
  ab_test → audiences                      (test_audience_ids, M:N)
  ab_test → products                       (tested_product_ids, M:N)
  ab_test → ab_test                        (parent_experiment_id, CHAIN)
  ab_test → segments                       (segment_ids_tested, M:N)

  loyalty → customers                      (member_id)
  loyalty → orders                         (related_order_id)
  loyalty → campaigns                      (related_campaign_id)
  loyalty → products                       (product_ids, M:N)
  loyalty → customers                      (referred_by_member_id, CIRCULAR)
  loyalty → customers                      (referral_chain_ids, M:N CIRCULAR)
  loyalty → crm_interactions               (triggered_by_interaction_id)

  content_assets → campaigns               (linked_campaign_ids, M:N)
  content_assets → products                (linked_product_ids, M:N)
  content_assets → influencers             (created_by_influencer)
  content_assets → content_assets          (parent/derived, CHAIN+M:N SELF)
  content_assets → ab_tests                (used_in_experiment_ids, M:N)
  content_assets → audiences               (target_audience_ids, M:N)
  content_assets → agencies                (producing_agency_id)

  DERIVED RELATIONSHIPS (require 3+ file joins):
  • customer → email → conversion → product (4 files)
  • customer → loyalty → campaign → influencer → content (5 files)
  • impression → campaign → deal → budget (4 files)
  • social_post → influencer → agency → budget (4 files)
  • ab_test → creative → campaign → customer_segment → customer (5 files)
  • crm_interaction → order → product → competitor_analysis (4 files)
""")

    print(f"\n  DIRTY DATA SUMMARY:")
    print(f"    • ~8% null/empty/N/A/nan/미입력 per column")
    print(f"    • ~3% typos in categorical columns")
    print(f"    • Mixed date formats WITHIN same column")
    print(f"    • Mixed boolean: 0/1/Y/N/true/false/동의/미동의")
    print(f"    • Mixed languages: 한국어/English/日本語")
    print(f"    • Mixed casing: active/ACTIVE/Active/활성")
    print(f"    • Orphan FKs: 5% CRM, 3% budget")
    print(f"    • Contradictory duplicates: 80 rows in conversion_funnel")
    print(f"    • Contradictory status/dates: ~8% in campaign_registry")
    print(f"    • Pipe-separated M:N IDs in 50+ columns")
    print(f"    • Self-referencing chains in 8 files")
    print(f"    • Circular references: customer referrals, loyalty chains")
    print(f"    • Currency silos: KRW / USD / JPY")
    print(f"    • Inconsistent exchange rates (influencer fee_krw vs fee_usd)")
    print()
