#!/usr/bin/env python3
"""Generate realistic beauty e-commerce silo dataset.

Simulates a mid-large beauty online retailer where data lives across
separate systems (ERP, CRM, PIM, WMS, Marketing Platform) that evolved independently.

10 files, ~50K rows total, ~1,500 cols, 40+ cross-file relationships.

SILO EFFECTS:
- 고객 ID: customer_id / member_no / user_uid / buyer_code / cust_key (각 시스템마다 다름)
- 상품 ID: product_id / sku / item_code / prd_no / goods_cd
- 주문 ID: order_id / order_no / ord_num / transaction_id
- 날짜 형식: YYYY-MM-DD / YYYYMMDD / DD/MM/YYYY / epoch 혼재
- 가격 통화: KRW (대부분) / USD (글로벌몰) / JPY (일본몰)
- Boolean: True/False / Y/N / 1/0 / yes/no 혼재

RELATIONSHIP COMPLEXITY:
- 주문→주문상세→상품→성분 (chain)
- 고객→리뷰→상품, 고객→주문→반품 (fan-out)
- 상품→카테고리 hierarchy (L1→L2→L3→L4)
- 상품→성분 (M:N), 프로모션→상품 (M:N)
- 번들/세트 상품→구성상품 (self M:N)
- 고객→추천인→고객 (circular)
- 재고→입고→공급사→브랜드→상품 (deep chain)
- 장바구니 이탈 → 주문 전환 (cross-file derived)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random
import string
import hashlib

np.random.seed(42)
random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "beauty_ecommerce")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Entity pools ───
N_CUST = 8000
N_PROD = 2000
N_ORDER = 15000
N_BRAND = 120
N_SUPPLIER = 60
N_WAREHOUSE = 5
N_PROMO = 200
N_INGREDIENT = 500
N_COUPON_CODE = 1000

# ─── Shared IDs ───
CUST_IDS = [f"CU{str(i).zfill(7)}" for i in range(1, N_CUST + 1)]
PROD_IDS = [f"PRD-{str(i).zfill(5)}" for i in range(1, N_PROD + 1)]
ORDER_IDS = [f"ORD{str(i).zfill(8)}" for i in range(1, N_ORDER + 1)]
SKU_IDS = [f"SKU{str(i).zfill(8)}" for i in range(1, N_PROD + 1)]
BRAND_IDS = [f"BR-{str(i).zfill(3)}" for i in range(1, N_BRAND + 1)]
SUPP_IDS = [f"SUP-{str(i).zfill(3)}" for i in range(1, N_SUPPLIER + 1)]
WH_IDS = [f"WH-{str(i).zfill(2)}" for i in range(1, N_WAREHOUSE + 1)]
PROMO_IDS = [f"PM-{str(i).zfill(4)}" for i in range(1, N_PROMO + 1)]
INGR_IDS = [f"ING-{str(i).zfill(4)}" for i in range(1, N_INGREDIENT + 1)]
COUPON_CODES = [f"BEAUTY{''.join(random.choices(string.ascii_uppercase + string.digits, k=6))}"
                for _ in range(N_COUPON_CODE)]

# ─── Product → SKU mapping (same product, different ID in different system) ───
PROD_SKU_MAP = dict(zip(PROD_IDS, SKU_IDS))

# ─── Customer ID aliases for silo effect ───
CUST_MEMBER_MAP = {c: f"M{hashlib.md5(c.encode()).hexdigest()[:8].upper()}" for c in CUST_IDS}
CUST_UID_MAP = {c: f"uid_{hashlib.sha1(c.encode()).hexdigest()[:12]}" for c in CUST_IDS}

# ─── Pre-built relationships (M:N) ───
# Brand → Supplier (M:1)
BRAND_SUPP = {b: random.choice(SUPP_IDS) for b in BRAND_IDS}
# Product → Brand (M:1)
PROD_BRAND = {p: random.choice(BRAND_IDS) for p in PROD_IDS}
# Product → Ingredients (M:N, 3~15 ingredients per product)
PROD_INGR = {p: random.sample(INGR_IDS, random.randint(3, 15)) for p in PROD_IDS}
# Promo → Products (M:N, 5~50 products per promo)
PROMO_PROD = {pr: random.sample(PROD_IDS, random.randint(5, min(50, N_PROD))) for pr in PROMO_IDS}
# Bundle: ~10% of products are bundles containing 2~5 other products
BUNDLE_PRODS = {}
bundle_candidates = random.sample(PROD_IDS, N_PROD // 10)
non_bundle = [p for p in PROD_IDS if p not in bundle_candidates]
for bp in bundle_candidates:
    BUNDLE_PRODS[bp] = random.sample(non_bundle, random.randint(2, 5))
# Customer → Referrer (circular, ~20% have referrers)
CUST_REFERRER = {}
for c in random.sample(CUST_IDS, N_CUST // 5):
    CUST_REFERRER[c] = random.choice([x for x in CUST_IDS if x != c])

# ─── Beauty domain constants ───
CATEGORIES_L1 = ["스킨케어", "메이크업", "헤어케어", "바디케어", "향수",
                 "네일", "남성", "더마", "선케어", "클렌징"]
CATEGORIES_L2 = {
    "스킨케어": ["토너", "세럼", "에센스", "크림", "아이크림", "앰플", "미스트", "오일", "팩/마스크", "필링"],
    "메이크업": ["파운데이션", "립스틱", "아이섀도", "마스카라", "블러셔", "컨실러", "프라이머", "세팅파우더", "아이라이너", "립글로스"],
    "헤어케어": ["샴푸", "컨디셔너", "트리트먼트", "헤어오일", "헤어에센스", "두피케어", "스타일링", "염색"],
    "바디케어": ["바디로션", "바디워시", "핸드크림", "풋크림", "바디스크럽", "바디오일", "데오도란트"],
    "향수": ["오드퍼퓸", "오드뚜왈렛", "코롱", "솔리드퍼퓸", "바디미스트", "디퓨저", "캔들"],
    "네일": ["네일폴리시", "젤네일", "네일케어", "네일리무버", "네일스티커"],
    "남성": ["면도", "올인원", "남성스킨케어", "남성향수", "남성헤어"],
    "더마": ["시카", "아크네", "민감성", "재생", "보습강화"],
    "선케어": ["선크림", "선스틱", "선스프레이", "선쿠션", "애프터선"],
    "클렌징": ["클렌징폼", "클렌징오일", "클렌징워터", "클렌징밤", "클렌징티슈"],
}
SKIN_TYPES = ["건성", "지성", "복합성", "민감성", "중성", "트러블성"]
SKIN_CONCERNS = ["모공", "주름", "미백", "보습", "트러블", "탄력", "다크서클",
                 "각질", "피지", "홍조", "색소침착", "아토피"]
BRANDS = ["이니스프리", "설화수", "라네즈", "에뛰드", "미샤", "토니모리", "더페이스샵",
          "네이처리퍼블릭", "스킨푸드", "클리오", "바닐라코", "홀리카홀리카",
          "코스알엑스", "닥터자르트", "아이오페", "헤라", "VDL", "에스쁘아",
          "아모레퍼시픽", "LG생건", "애경", "올리브영PB", "무신사뷰티",
          "조선미녀", "넘버즈", "달바", "메디힐", "파파레시피", "아누아", "라운드랩",
          "구달", "아비브", "토리든", "일리윤", "세타필", "라로슈포제",
          "에스티로더", "랑콤", "맥", "바비브라운", "샤넬", "디올",
          "나스", "어반디케이", "투쿨포스쿨", "롬앤", "페리페라", "데이지크"]
PAYMENT_METHODS = ["신용카드", "체크카드", "네이버페이", "카카오페이", "토스페이",
                   "무통장입금", "휴대폰결제", "포인트전액", "삼성페이", "페이코"]
SHIPPING_METHODS = ["일반배송", "로켓배송", "새벽배송", "퀵배송", "편의점픽업",
                    "오늘출발", "해외직배송", "묶음배송"]
PLATFORMS = ["PC웹", "모바일웹", "iOS앱", "Android앱", "카카오톡채널",
             "네이버스마트스토어", "쿠팡", "올리브영온라인", "무신사"]
RETURN_REASONS = ["피부트러블", "색상불일치", "용량부족", "유통기한임박", "변심",
                  "파손/불량", "오배송", "알러지반응", "향불만", "가품의심",
                  "사이즈맞지않음", "성분불만", "텍스처불만"]
INGREDIENTS_LIST = [
    "히알루론산", "나이아신아마이드", "레티놀", "비타민C", "세라마이드", "펩타이드",
    "AHA", "BHA", "PHA", "살리실산", "글리콜산", "아젤라인산",
    "티트리오일", "알로에", "녹차추출물", "프로폴리스", "꿀추출물", "달팽이뮤신",
    "콜라겐", "엘라스틴", "스쿠알란", "호호바오일", "아르간오일", "시어버터",
    "판테놀", "알란토인", "마데카소사이드", "병풀추출물", "감초추출물", "어성초",
    "로즈힙오일", "카밀러", "라벤더", "유칼립투스", "백단향",
    "징크옥사이드", "티타늄디옥사이드", "토코페롤", "아데노신", "카페인",
    "바쿠치올", "글루타치온", "트라넥삼산", "아스코빌글루코사이드", "알부틴",
]

dates_range = pd.date_range("2023-01-01", "2025-12-31", freq="D")


# ─── Helpers ───
def uid():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def rdates(n, start="2023-01-01", end="2025-12-31"):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return [pd.Timestamp(s + (e - s) * random.random()) for _ in range(n)]


def dirty(series, null_pct=0.06, typo_pct=0.02):
    """Inject nulls and typos into a series."""
    s = series.copy()
    n = len(s)
    # nulls
    null_idx = np.random.choice(n, int(n * null_pct), replace=False)
    s.iloc[null_idx] = np.nan
    # typos for string cols
    if pd.api.types.is_object_dtype(s):
        valid = s.dropna().index
        if len(valid) > 0:
            typo_idx = np.random.choice(valid, min(int(n * typo_pct), len(valid)), replace=False)
            for idx in typo_idx:
                v = str(s.iloc[idx])
                if len(v) > 2:
                    pos = random.randint(0, len(v) - 2)
                    s.iloc[idx] = v[:pos] + random.choice("!@#_") + v[pos + 1:]
    return s


def mixed_dates(dates_list):
    """Same column, different date formats → silo nightmare."""
    result = []
    for d in dates_list:
        if pd.isna(d):
            result.append(None)
            continue
        ts = pd.Timestamp(d)
        fmt = random.choice(["%Y-%m-%d", "%Y%m%d", "%d/%m/%Y", "%m-%d-%Y", "%Y.%m.%d"])
        result.append(ts.strftime(fmt))
    return result


def multi_id(id_list, min_k=1, max_k=4):
    """Pipe-separated multi-value for M:N in single column."""
    k = random.randint(min_k, min(max_k, len(id_list)))
    return "|".join(random.sample(id_list, k))


def bulk_metrics(prefix, n, count=30):
    """Generate bulk numeric metric columns."""
    cols = {}
    for i in range(count):
        col_name = f"{prefix}_{i+1:02d}"
        cols[col_name] = np.random.exponential(scale=random.uniform(10, 1000), size=n).round(2)
    return cols


def bulk_flags(prefix, n, count=15):
    """Generate bulk boolean-ish flag columns with mixed formats."""
    cols = {}
    formats = [
        lambda: random.choice(["Y", "N"]),
        lambda: random.choice(["True", "False"]),
        lambda: random.choice([1, 0]),
        lambda: random.choice(["yes", "no"]),
        lambda: random.choice(["O", "X"]),
    ]
    for i in range(count):
        fmt = random.choice(formats)
        cols[f"{prefix}_{i+1:02d}"] = [fmt() for _ in range(n)]
    return cols


def bulk_scores(prefix, n, count=20):
    """Generate bulk score columns (0~100)."""
    cols = {}
    for i in range(count):
        cols[f"{prefix}_{i+1:02d}"] = np.random.beta(2, 5, size=n).round(4) * 100
    return cols


# ═══════════════════════════════════════════════════════════════════
# TABLE 1: 상품 마스터 (PIM System)
# ═══════════════════════════════════════════════════════════════════
def gen_products():
    n = N_PROD
    cats_l1 = np.random.choice(CATEGORIES_L1, n)
    cats_l2 = [random.choice(CATEGORIES_L2.get(c, ["기타"])) for c in cats_l1]

    data = {
        "product_id": PROD_IDS,
        "sku_code": SKU_IDS,
        "barcode": [f"880{random.randint(1000000000, 9999999999)}" for _ in range(n)],
        "product_name_ko": [f"{random.choice(BRANDS)} {random.choice(['리뉴얼','더블','슈퍼','울트라','프로','맥스','퓨어','센시티브'])} {c2} {random.choice(['50ml','100ml','200ml','30g','50g','#01','#02','#03','세트','기획세트','리필'])}" for c2 in cats_l2],
        "product_name_en": [f"Product_{uid()}" for _ in range(n)],
        "brand_id": [PROD_BRAND[p] for p in PROD_IDS],
        "brand_name": [random.choice(BRANDS) for _ in range(n)],  # intentionally may mismatch brand_id
        "category_l1": cats_l1,
        "category_l2": cats_l2,
        "category_l3": [f"{c2}_{random.choice(['베이직','프리미엄','더마','한방','비건','오가닉','남성용','시즌한정'])}" for c2 in cats_l2],
        "category_l4": [f"sub_{uid()}" for _ in range(n)],
        "retail_price_krw": np.random.choice([9900, 12000, 15000, 18000, 22000, 25000, 28000,
                                               32000, 38000, 45000, 52000, 68000, 89000, 120000,
                                               150000, 198000, 250000, 350000], n),
        "cost_price_krw": np.random.randint(3000, 100000, n),
        "global_price_usd": np.round(np.random.uniform(5, 150, n), 2),
        "jp_price_jpy": np.random.choice([990, 1480, 1980, 2480, 2980, 3480, 3980, 4980, 6980, 9800, 14800], n),
        "weight_g": np.random.choice([15, 30, 50, 75, 100, 150, 200, 250, 300, 500, 1000], n),
        "volume_ml": np.random.choice([5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 250, 300, 500], n),
        "is_bundle": ["Y" if p in BUNDLE_PRODS else "N" for p in PROD_IDS],
        "bundle_components": ["|".join(BUNDLE_PRODS.get(p, [])) for p in PROD_IDS],
        "ingredient_ids": ["|".join(PROD_INGR[p]) for p in PROD_IDS],
        "key_ingredient_1": [random.choice(INGREDIENTS_LIST) for _ in range(n)],
        "key_ingredient_2": [random.choice(INGREDIENTS_LIST) for _ in range(n)],
        "key_ingredient_3": [random.choice(INGREDIENTS_LIST) for _ in range(n)],
        "skin_type_target": ["|".join(random.sample(SKIN_TYPES, random.randint(1, 3))) for _ in range(n)],
        "skin_concern_target": ["|".join(random.sample(SKIN_CONCERNS, random.randint(1, 4))) for _ in range(n)],
        "launch_date": mixed_dates(rdates(n, "2020-01-01", "2025-06-30")),
        "discontinue_date": [mixed_dates([rdates(1, "2024-01-01", "2026-12-31")[0]])[0]
                             if random.random() < 0.1 else None for _ in range(n)],
        "status": np.random.choice(["판매중", "일시품절", "단종", "예약판매", "시즌한정", "입고예정"], n,
                                   p=[0.65, 0.1, 0.08, 0.05, 0.07, 0.05]),
        "vegan_yn": np.random.choice(["Y", "N", "해당없음"], n, p=[0.15, 0.7, 0.15]),
        "cruelty_free": np.random.choice([True, False, "인증중"], n, p=[0.2, 0.65, 0.15]),
        "ewa_grade": np.random.choice(["1", "2", "3", "4", "5", "미검증", None], n,
                                      p=[0.1, 0.2, 0.25, 0.15, 0.05, 0.15, 0.1]),
        "origin_country": np.random.choice(["한국", "프랑스", "일본", "미국", "독일", "이탈리아", "호주", "태국"], n,
                                           p=[0.5, 0.12, 0.1, 0.08, 0.06, 0.05, 0.05, 0.04]),
        "shelf_life_months": np.random.choice([6, 12, 18, 24, 30, 36], n),
        "supplier_id": [BRAND_SUPP[PROD_BRAND[p]] for p in PROD_IDS],
        "min_order_qty": np.random.choice([1, 6, 12, 24, 48, 100], n),
        "tax_type": np.random.choice(["과세", "면세", "영세"], n, p=[0.8, 0.15, 0.05]),
    }

    # Bulk attribute columns (PIM systems love these)
    for i in range(40):
        data[f"attr_{i+1:02d}"] = [f"val_{random.randint(1, 50)}" if random.random() > 0.3 else None
                                    for _ in range(n)]
    # Bulk scores
    data.update(bulk_scores("quality_score", n, 20))
    data.update(bulk_metrics("pim_metric", n, 25))
    data.update(bulk_flags("pim_flag", n, 15))

    df = pd.DataFrame(data)
    for col in ["product_name_ko", "brand_name", "category_l3"]:
        df[col] = dirty(df[col])
    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 2: 고객 마스터 (CRM System)
# ═══════════════════════════════════════════════════════════════════
def gen_customers():
    n = N_CUST
    data = {
        "customer_id": CUST_IDS,
        "member_no": [CUST_MEMBER_MAP[c] for c in CUST_IDS],  # CRM system ID
        "user_uid": [CUST_UID_MAP[c] for c in CUST_IDS],  # web/app system ID
        "name": [f"{'김이박최정강조윤장임'[random.randint(0,9)]}{'가나다라마바사아자차카타파하'[random.randint(0,13)]}{'영미진수현준서지예은호민'[random.randint(0,11)]}" for _ in range(n)],
        "email": [f"user_{uid()}@{random.choice(['gmail.com','naver.com','kakao.com','daum.net','hanmail.net','nate.com','outlook.com'])}" for _ in range(n)],
        "phone": [f"010-{random.randint(1000,9999)}-{random.randint(1000,9999)}" for _ in range(n)],
        "birth_date": mixed_dates(rdates(n, "1970-01-01", "2005-12-31")),
        "gender": np.random.choice(["F", "M", "여", "남", None], n, p=[0.45, 0.25, 0.1, 0.05, 0.15]),
        "age_group": np.random.choice(["10대", "20대초반", "20대후반", "30대초반", "30대후반",
                                       "40대", "50대", "60대이상"], n),
        "skin_type": np.random.choice(SKIN_TYPES, n),
        "skin_concern_1": np.random.choice(SKIN_CONCERNS, n),
        "skin_concern_2": [random.choice(SKIN_CONCERNS) if random.random() > 0.3 else None for _ in range(n)],
        "skin_concern_3": [random.choice(SKIN_CONCERNS) if random.random() > 0.6 else None for _ in range(n)],
        "membership_tier": np.random.choice(["WELCOME", "SILVER", "GOLD", "VIP", "VVIP", "블랙"], n,
                                            p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05]),
        "tier_updated_at": mixed_dates(rdates(n)),
        "signup_date": mixed_dates(rdates(n, "2019-01-01", "2025-12-31")),
        "signup_channel": np.random.choice(PLATFORMS, n),
        "last_login": mixed_dates(rdates(n, "2024-01-01", "2025-12-31")),
        "total_orders": np.random.poisson(8, n),
        "total_spent_krw": np.random.exponential(200000, n).astype(int),
        "total_points": np.random.exponential(5000, n).astype(int),
        "available_points": np.random.exponential(2000, n).astype(int),
        "referrer_id": [CUST_REFERRER.get(c, None) for c in CUST_IDS],
        "preferred_brand_1": [random.choice(BRANDS) for _ in range(n)],
        "preferred_brand_2": [random.choice(BRANDS) if random.random() > 0.3 else None for _ in range(n)],
        "preferred_category": np.random.choice(CATEGORIES_L1, n),
        "marketing_consent": np.random.choice(["Y", "N", True, False, 1, 0], n),
        "sms_consent": np.random.choice(["동의", "미동의", "Y", "N"], n),
        "email_consent": np.random.choice([True, False], n),
        "push_consent": np.random.choice(["Y", "N"], n),
        "addr_city": np.random.choice(["서울", "경기", "인천", "부산", "대구", "대전", "광주",
                                        "울산", "세종", "강원", "충북", "충남", "전북", "전남",
                                        "경북", "경남", "제주"], n),
        "addr_zip": [f"{random.randint(10000, 63999)}" for _ in range(n)],
        "is_dormant": np.random.choice(["Y", "N"], n, p=[0.15, 0.85]),
        "is_blacklist": np.random.choice([0, 1], n, p=[0.97, 0.03]),
        "ltv_segment": np.random.choice(["상위1%", "상위5%", "상위10%", "상위20%", "중간", "하위", "신규"], n),
        "rfm_r_score": np.random.randint(1, 6, n),
        "rfm_f_score": np.random.randint(1, 6, n),
        "rfm_m_score": np.random.randint(1, 6, n),
        "rfm_segment": np.random.choice(["챔피언", "충성고객", "잠재충성", "신규고객",
                                          "유망고객", "관심필요", "이탈직전", "수면고객", "이탈고객"], n),
        "beauty_profile_id": [f"BP-{uid()}" for _ in range(n)],
    }

    # Bulk CRM attributes
    data.update(bulk_scores("crm_score", n, 25))
    data.update(bulk_metrics("crm_metric", n, 20))
    data.update(bulk_flags("crm_flag", n, 15))
    # Beauty preference tags
    for i in range(20):
        data[f"beauty_pref_{i+1:02d}"] = [random.choice(INGREDIENTS_LIST + SKIN_CONCERNS + CATEGORIES_L1)
                                           if random.random() > 0.4 else None for _ in range(n)]

    df = pd.DataFrame(data)
    for col in ["name", "email", "addr_city"]:
        df[col] = dirty(df[col])
    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 3: 주문 (ERP/OMS)
# ═══════════════════════════════════════════════════════════════════
def gen_orders():
    n = N_ORDER
    order_dates = rdates(n)
    custs = np.random.choice(CUST_IDS, n)

    data = {
        "order_no": ORDER_IDS,
        "transaction_id": [f"TXN-{uid().upper()}" for _ in range(n)],
        # silo: CRM uses customer_id, OMS uses buyer_code
        "buyer_code": [CUST_MEMBER_MAP[c] for c in custs],
        "order_date": mixed_dates(order_dates),
        "order_timestamp": [int(pd.Timestamp(d).timestamp()) if random.random() < 0.3
                            else pd.Timestamp(d).strftime("%Y-%m-%d %H:%M:%S") for d in order_dates],
        "order_status": np.random.choice(["완료", "배송중", "결제완료", "취소", "반품접수",
                                           "환불완료", "교환중", "부분취소"], n,
                                          p=[0.55, 0.12, 0.08, 0.08, 0.05, 0.05, 0.04, 0.03]),
        "payment_method": np.random.choice(PAYMENT_METHODS, n),
        "payment_amount_krw": np.random.exponential(45000, n).astype(int),
        "discount_amount": np.random.exponential(5000, n).astype(int),
        "coupon_code": [random.choice(COUPON_CODES) if random.random() < 0.4 else None for _ in range(n)],
        "coupon_discount": [random.choice([1000, 2000, 3000, 5000, 10000, 15000, 20000])
                            if random.random() < 0.4 else 0 for _ in range(n)],
        "point_used": np.random.exponential(500, n).astype(int),
        "point_earned": np.random.exponential(300, n).astype(int),
        "shipping_method": np.random.choice(SHIPPING_METHODS, n),
        "shipping_fee": np.random.choice([0, 0, 0, 2500, 3000, 3500, 5000], n),
        "shipping_addr_zip": [f"{random.randint(10000, 63999)}" for _ in range(n)],
        "estimated_delivery": mixed_dates([d + timedelta(days=random.randint(1, 7)) for d in order_dates]),
        "actual_delivery": [mixed_dates([d + timedelta(days=random.randint(1, 10))])[0]
                            if random.random() < 0.7 else None for d in order_dates],
        "platform": np.random.choice(PLATFORMS, n),
        "device_type": np.random.choice(["mobile", "desktop", "tablet", "app_ios", "app_android"], n,
                                         p=[0.35, 0.2, 0.05, 0.25, 0.15]),
        "is_first_order": np.random.choice(["Y", "N", True, False], n, p=[0.15, 0.55, 0.15, 0.15]),
        "is_gift": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "gift_message": [f"선물_{uid()}" if random.random() < 0.1 else None for _ in range(n)],
        "promo_id": [random.choice(PROMO_IDS) if random.random() < 0.35 else None for _ in range(n)],
        "utm_source": np.random.choice(["naver", "google", "kakao", "instagram", "youtube",
                                         "direct", "affiliate", None], n),
        "utm_medium": np.random.choice(["cpc", "organic", "social", "email", "referral", None], n),
        "utm_campaign": [f"camp_{uid()}" if random.random() > 0.3 else None for _ in range(n)],
        "session_id": [f"sess_{uid()}" for _ in range(n)],  # links to web_events
    }

    # Bulk order metrics
    data.update(bulk_metrics("order_metric", n, 20))
    data.update(bulk_flags("order_flag", n, 10))

    df = pd.DataFrame(data)
    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 4: 주문상세 (ERP/OMS — line items)
# ═══════════════════════════════════════════════════════════════════
def gen_order_items():
    rows = []
    for oid in ORDER_IDS:
        n_items = np.random.choice([1, 1, 1, 2, 2, 3, 4, 5])
        prods = random.sample(PROD_IDS, n_items)
        for seq, pid in enumerate(prods, 1):
            sku = PROD_SKU_MAP[pid]
            qty = random.choice([1, 1, 1, 2, 2, 3])
            unit_price = random.choice([9900, 12000, 15000, 19800, 22000, 25000, 29000,
                                         32000, 38000, 45000, 52000, 68000, 89000])
            rows.append({
                "order_no": oid,
                "line_seq": seq,
                # silo: uses item_code instead of product_id
                "item_code": sku,
                "goods_cd": pid,  # another system's ID for same product
                "quantity": qty,
                "unit_price_krw": unit_price,
                "line_total_krw": unit_price * qty,
                "discount_rate": random.choice([0, 0, 0, 5, 10, 15, 20, 25, 30, 40, 50]),
                "final_price": int(unit_price * qty * (1 - random.choice([0, 0.05, 0.1, 0.15, 0.2, 0.3]))),
                "is_freebie": random.choice(["N", "N", "N", "N", "Y"]),
                "gift_wrap": random.choice([0, 0, 0, 1]),
                "option_color": random.choice(["#01 로즈", "#02 코랄", "#03 베이지", "#04 핑크",
                                                 "#05 레드", "N/A", None]),
                "option_size": random.choice(["미니", "레귤러", "라지", "점보", "여행용", "N/A", None]),
            })

    n = len(rows)
    df = pd.DataFrame(rows)

    # Bulk line item metrics
    for i in range(20):
        df[f"item_attr_{i+1:02d}"] = [f"v{random.randint(1,30)}" if random.random() > 0.4 else None
                                       for _ in range(n)]
    met = bulk_metrics("item_metric", n, 15)
    for k, v in met.items():
        df[k] = v
    fl = bulk_flags("item_flag", n, 10)
    for k, v in fl.items():
        df[k] = v

    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 5: 리뷰 (UGC Platform)
# ═══════════════════════════════════════════════════════════════════
def gen_reviews():
    n = 6000
    data = {
        "review_id": [f"RV-{str(i).zfill(7)}" for i in range(1, n + 1)],
        # silo: uses user_uid instead of customer_id
        "user_uid": [CUST_UID_MAP[random.choice(CUST_IDS)] for _ in range(n)],
        "nickname": [f"뷰티{''.join(random.choices('가나다라마바사아자차', k=2))}{random.randint(1,999)}" for _ in range(n)],
        # silo: uses goods_cd
        "goods_cd": np.random.choice(PROD_IDS, n),
        "order_ref": [random.choice(ORDER_IDS) if random.random() < 0.85 else None for _ in range(n)],
        "rating": np.random.choice([1, 2, 3, 4, 5], n, p=[0.03, 0.05, 0.12, 0.35, 0.45]),
        "title": [f"{'좋아요 너무좋아요 추천합니다 재구매의사있음 그냥그래요 별로에요 실망 최고 대박 가성비'.split()[random.randint(0,9)]}" for _ in range(n)],
        "content": [f"리뷰내용_{uid()}_{'피부가좋아졌어요 향이좋아요 발림성이좋아요 보습력최고 자극없어요 트러블났어요 가성비좋아요 재구매예정'.split()[random.randint(0,7)]}" for _ in range(n)],
        "review_date": mixed_dates(rdates(n)),
        "verified_purchase": np.random.choice(["Y", "N", True, False, 1, 0], n),
        "photo_count": np.random.choice([0, 0, 0, 0, 1, 1, 2, 3, 5], n),
        "video_yn": np.random.choice(["Y", "N"], n, p=[0.05, 0.95]),
        "helpful_count": np.random.poisson(3, n),
        "report_count": np.random.choice([0, 0, 0, 0, 0, 1, 2], n),
        "reviewer_skin_type": np.random.choice(SKIN_TYPES + [None], n),
        "reviewer_age_group": np.random.choice(["10대", "20대", "30대", "40대", "50대이상", None], n),
        "sentiment_score": np.round(np.random.beta(5, 2, n) * 100, 1),
        "admin_reply": [f"답변_{uid()}" if random.random() < 0.2 else None for _ in range(n)],
        "is_best": np.random.choice(["Y", "N"], n, p=[0.05, 0.95]),
    }

    data.update(bulk_scores("review_nlp", n, 15))
    data.update(bulk_flags("review_flag", n, 10))

    df = pd.DataFrame(data)
    df["content"] = dirty(df["content"], null_pct=0.03)
    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 6: 재고/물류 (WMS)
# ═══════════════════════════════════════════════════════════════════
def gen_inventory():
    rows = []
    for pid in PROD_IDS:
        for wh in random.sample(WH_IDS, random.randint(1, 3)):
            rows.append({
                # silo: WMS uses sku, not product_id
                "sku": PROD_SKU_MAP[pid],
                "warehouse_id": wh,
                "warehouse_name": f"{'김포|인천|부산|대구|광주'.split('|')[int(wh[-2:])-1]}센터",
                "location_code": f"{random.choice('ABCDEF')}-{random.randint(1,50):02d}-{random.randint(1,5)}",
                "current_stock": random.randint(0, 500),
                "reserved_stock": random.randint(0, 50),
                "available_stock": random.randint(0, 450),
                "safety_stock": random.choice([10, 20, 30, 50, 100]),
                "reorder_point": random.choice([20, 30, 50, 80, 100, 150]),
                "max_stock": random.choice([200, 300, 500, 800, 1000]),
                "last_inbound_date": mixed_dates(rdates(1))[0],
                "last_outbound_date": mixed_dates(rdates(1))[0],
                "lot_number": f"LOT{random.randint(20230101, 20251231)}",
                "expiry_date": mixed_dates(rdates(1, "2025-01-01", "2027-12-31"))[0],
                "manufacture_date": mixed_dates(rdates(1, "2023-01-01", "2025-06-30"))[0],
                "inbound_cost_krw": random.randint(1000, 50000),
                "storage_fee_daily": round(random.uniform(10, 100), 2),
                "status": random.choice(["정상", "정상", "정상", "이상재고", "반품입고", "폐기예정"]),
                "temperature_zone": random.choice(["상온", "상온", "상온", "냉장", "냉동"]),
                "supplier_ref": BRAND_SUPP[PROD_BRAND[pid]],
            })

    n = len(rows)
    df = pd.DataFrame(rows)

    data_extra = bulk_metrics("wms_metric", n, 20)
    for k, v in data_extra.items():
        df[k] = v
    fl = bulk_flags("wms_flag", n, 10)
    for k, v in fl.items():
        df[k] = v

    df["sku"] = dirty(df["sku"], null_pct=0.02, typo_pct=0.01)
    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 7: 프로모션/쿠폰 (Marketing Platform)
# ═══════════════════════════════════════════════════════════════════
def gen_promotions():
    n = N_PROMO
    start_dates = rdates(n, "2023-06-01", "2025-10-31")
    data = {
        "promo_id": PROMO_IDS,
        "promo_name": [f"{'봄세일 여름특가 가을감사 겨울빅세일 블프 더블적립 신상런칭 회원감사 앵콜특가 1+1 타임딜 브랜드위크'.split()[random.randint(0,11)]}_{uid()[:4]}" for _ in range(n)],
        "promo_type": np.random.choice(["할인율", "정액할인", "1+1", "증정", "적립금", "무료배송",
                                         "세트할인", "첫구매", "재구매", "생일"], n),
        "discount_type": np.random.choice(["percent", "fixed", "bogo", "gift"], n),
        "discount_value": np.random.choice([5, 10, 15, 20, 25, 30, 40, 50, 1000, 2000, 3000, 5000, 10000], n),
        "min_purchase_amount": np.random.choice([0, 10000, 20000, 30000, 50000, 100000], n),
        "max_discount_cap": np.random.choice([0, 5000, 10000, 20000, 50000, 100000], n),
        "start_date": mixed_dates(start_dates),
        "end_date": mixed_dates([d + timedelta(days=random.randint(3, 60)) for d in start_dates]),
        "status": np.random.choice(["진행중", "종료", "예정", "일시중지"], n, p=[0.3, 0.45, 0.15, 0.1]),
        "target_segment": ["|".join(random.sample(["VIP", "VVIP", "신규", "이탈방지", "전체",
                                                    "첫구매", "재구매", "생일고객"], random.randint(1, 3)))
                           for _ in range(n)],
        "target_products": ["|".join(PROMO_PROD[p]) for p in PROMO_IDS],
        "target_categories": ["|".join(random.sample(CATEGORIES_L1, random.randint(1, 4))) for _ in range(n)],
        "target_brands": ["|".join(random.sample(BRANDS, random.randint(1, 5))) for _ in range(n)],
        "channel": ["|".join(random.sample(PLATFORMS, random.randint(1, 3))) for _ in range(n)],
        "budget_krw": np.random.choice([500000, 1000000, 3000000, 5000000, 10000000, 50000000], n),
        "used_budget_krw": np.random.exponential(2000000, n).astype(int),
        "redemption_count": np.random.poisson(200, n),
        "coupon_codes": ["|".join(random.sample(COUPON_CODES, random.randint(1, 5))) for _ in range(n)],
        "stacking_allowed": np.random.choice(["Y", "N"], n, p=[0.2, 0.8]),
        "created_by": [f"admin_{random.randint(1,20)}" for _ in range(n)],
    }

    data.update(bulk_metrics("promo_metric", n, 15))
    data.update(bulk_flags("promo_flag", n, 10))

    df = pd.DataFrame(data)
    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 8: 웹/앱 이벤트 로그 (Analytics)
# ═══════════════════════════════════════════════════════════════════
def gen_web_events():
    n = 10000
    event_dates = rdates(n, "2024-06-01", "2025-12-31")
    custs = [random.choice(CUST_IDS) if random.random() < 0.7 else None for _ in range(n)]

    data = {
        "event_id": [f"EVT-{str(i).zfill(8)}" for i in range(1, n + 1)],
        "session_id": [f"sess_{uid()}" for _ in range(n)],
        # silo: analytics uses user_uid
        "user_uid": [CUST_UID_MAP[c] if c else f"anon_{uid()}" for c in custs],
        "event_type": np.random.choice(["page_view", "product_view", "add_to_cart", "remove_cart",
                                         "begin_checkout", "purchase", "search", "filter",
                                         "wishlist_add", "wishlist_remove", "review_click",
                                         "share", "compare", "notification_click"], n,
                                        p=[0.25, 0.2, 0.12, 0.03, 0.05, 0.04, 0.1, 0.06,
                                           0.04, 0.01, 0.03, 0.02, 0.02, 0.03]),
        "event_timestamp": [int(pd.Timestamp(d).timestamp()) for d in event_dates],
        "page_url": [f"/{'category product search cart checkout mypage brand event'.split()[random.randint(0,7)]}/{uid()}" for _ in range(n)],
        # silo: uses prd_no
        "prd_no": [random.choice(PROD_IDS) if random.random() < 0.6 else None for _ in range(n)],
        "search_query": [random.choice(["세럼", "선크림", "쿠션", "클렌징", "토너", "마스크팩",
                                          "레티놀", "비타민C", "수분크림", None, None, None]) for _ in range(n)],
        "referrer": np.random.choice(["naver", "google", "instagram", "kakao", "direct",
                                       "youtube", "affiliate", None], n),
        "platform": np.random.choice(PLATFORMS, n),
        "device": np.random.choice(["mobile", "desktop", "tablet"], n, p=[0.6, 0.3, 0.1]),
        "os": np.random.choice(["iOS", "Android", "Windows", "macOS"], n, p=[0.35, 0.3, 0.2, 0.15]),
        "browser": np.random.choice(["Chrome", "Safari", "Samsung", "Edge", "Whale"], n),
        "screen_resolution": np.random.choice(["1080x2400", "1170x2532", "1920x1080",
                                                 "2560x1440", "390x844"], n),
        "time_on_page_sec": np.random.exponential(30, n).round(1),
        "scroll_depth_pct": np.random.uniform(0, 100, n).round(1),
        "cart_value_krw": [random.choice([0, 0, 0, 15000, 25000, 38000, 52000, 89000])
                           for _ in range(n)],
        "ab_test_variant": [random.choice(["A", "B", "C", "control", None]) for _ in range(n)],
    }

    data.update(bulk_metrics("analytics_metric", n, 20))
    data.update(bulk_flags("analytics_flag", n, 10))

    df = pd.DataFrame(data)
    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 9: 반품/환불 (CS System)
# ═══════════════════════════════════════════════════════════════════
def gen_returns():
    n = 3000
    data = {
        "return_id": [f"RT-{str(i).zfill(6)}" for i in range(1, n + 1)],
        # silo: CS system uses ord_num
        "ord_num": np.random.choice(ORDER_IDS, n),
        # silo: CS uses cust_key
        "cust_key": [CUST_MEMBER_MAP[random.choice(CUST_IDS)] for _ in range(n)],
        "return_type": np.random.choice(["반품", "교환", "부분반품", "전체반품"], n,
                                         p=[0.5, 0.25, 0.15, 0.1]),
        "return_reason": np.random.choice(RETURN_REASONS, n),
        "return_reason_detail": [f"상세사유_{uid()}" if random.random() < 0.5 else None for _ in range(n)],
        "request_date": mixed_dates(rdates(n)),
        "pickup_date": [mixed_dates(rdates(1))[0] if random.random() < 0.7 else None for _ in range(n)],
        "received_date": [mixed_dates(rdates(1))[0] if random.random() < 0.6 else None for _ in range(n)],
        "inspection_result": np.random.choice(["승인", "거절", "부분승인", "검수중", None], n,
                                               p=[0.5, 0.1, 0.1, 0.15, 0.15]),
        "refund_amount_krw": np.random.exponential(30000, n).astype(int),
        "refund_method": np.random.choice(["원결제수단", "포인트", "계좌이체", "쿠폰발급"], n),
        "refund_status": np.random.choice(["완료", "처리중", "대기", "거절"], n,
                                           p=[0.55, 0.2, 0.15, 0.1]),
        "refund_date": [mixed_dates(rdates(1))[0] if random.random() < 0.6 else None for _ in range(n)],
        "cs_agent_id": [f"agent_{random.randint(1, 30)}" for _ in range(n)],
        "cs_channel": np.random.choice(["채팅", "전화", "이메일", "카카오톡", "게시판"], n),
        "satisfaction_score": np.random.choice([1, 2, 3, 4, 5, None], n),
        # silo: uses item_code (SKU) for the returned item
        "returned_item_code": [PROD_SKU_MAP[random.choice(PROD_IDS)] for _ in range(n)],
        "returned_qty": np.random.choice([1, 1, 1, 2, 3], n),
        "photo_submitted": np.random.choice(["Y", "N"], n, p=[0.3, 0.7]),
        "is_repeat_returner": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    }

    data.update(bulk_metrics("cs_metric", n, 15))
    data.update(bulk_flags("cs_flag", n, 10))

    df = pd.DataFrame(data)
    return df


# ═══════════════════════════════════════════════════════════════════
# TABLE 10: 공급사/입고 (SCM/Procurement)
# ═══════════════════════════════════════════════════════════════════
def gen_suppliers():
    # Supplier master + recent inbound records
    rows = []
    for sid in SUPP_IDS:
        # ~30 inbound records per supplier
        n_records = random.randint(15, 50)
        brands_for_supp = [b for b, s in BRAND_SUPP.items() if s == sid]
        prods_for_supp = [p for p in PROD_IDS if PROD_BRAND[p] in brands_for_supp]
        if not prods_for_supp:
            prods_for_supp = random.sample(PROD_IDS, 5)

        for _ in range(n_records):
            pid = random.choice(prods_for_supp)
            inb_date = rdates(1)[0]
            rows.append({
                "supplier_id": sid,
                "supplier_name": f"{'주식회사 유한회사 (주)'.split()[random.randint(0,2)]} {random.choice(BRANDS)}코스메틱",
                "business_reg_no": f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10000,99999)}",
                "contact_person": f"{'김이박최정'[random.randint(0,4)]}{'담당자'}",
                "contact_email": f"contact_{uid()}@{random.choice(['supplier.co.kr','beauty.com','cosme.kr'])}",
                "inbound_id": f"INB-{uid().upper()}",
                "po_number": f"PO-{random.randint(20230001, 20259999)}",
                # silo: SCM uses sku
                "sku": PROD_SKU_MAP[pid],
                "brand_id": PROD_BRAND[pid],
                "inbound_qty": random.choice([50, 100, 200, 300, 500, 1000, 2000]),
                "unit_cost_krw": random.randint(2000, 80000),
                "total_cost_krw": random.randint(100000, 50000000),
                "inbound_date": mixed_dates([inb_date])[0],
                "expected_date": mixed_dates([inb_date - timedelta(days=random.randint(-5, 5))])[0],
                "warehouse_dest": random.choice(WH_IDS),
                "quality_check": random.choice(["합격", "합격", "합격", "부분합격", "불합격", "검사중"]),
                "defect_rate_pct": round(random.uniform(0, 5), 2),
                "lead_time_days": random.randint(3, 45),
                "payment_terms": random.choice(["선불", "30일", "60일", "90일", "위탁"]),
                "currency": random.choice(["KRW", "KRW", "KRW", "USD", "JPY", "EUR"]),
                "origin_country": random.choice(["한국", "중국", "프랑스", "일본", "미국", "독일"]),
            })

    n = len(rows)
    df = pd.DataFrame(rows)

    data_extra = bulk_metrics("scm_metric", n, 15)
    for k, v in data_extra.items():
        df[k] = v
    fl = bulk_flags("scm_flag", n, 8)
    for k, v in fl.items():
        df[k] = v

    df["supplier_name"] = dirty(df["supplier_name"])
    return df


# ═══════════════════════════════════════════════════════════════════
# Chaos injection (cross-file inconsistencies)
# ═══════════════════════════════════════════════════════════════════
def inject_chaos(dfs):
    """Inject cross-file inconsistencies for extra silo realism."""
    # 1. Duplicate some customer rows with slightly different data
    cust_df = dfs["customer_master"]
    dup_idx = np.random.choice(cust_df.index, 80, replace=False)
    dups = cust_df.loc[dup_idx].copy()
    dups["membership_tier"] = np.random.choice(["WELCOME", "SILVER", "GOLD", "VIP"], len(dups))
    dups["total_spent_krw"] = dups["total_spent_krw"] + np.random.randint(-50000, 50000, len(dups))
    dfs["customer_master"] = pd.concat([cust_df, dups], ignore_index=True)

    # 2. Some products in orders reference non-existent SKUs
    items_df = dfs["order_items"]
    orphan_idx = np.random.choice(items_df.index, 50, replace=False)
    items_df.loc[orphan_idx, "item_code"] = [f"SKU_ORPHAN_{uid()}" for _ in range(50)]
    dfs["order_items"] = items_df

    # 3. Price mismatch: same product different price in product_catalog vs order_items
    # (already somewhat natural, but we make it worse)
    prod_df = dfs["product_catalog"]
    mismatch_idx = np.random.choice(prod_df.index, 30, replace=False)
    prod_df.loc[mismatch_idx, "retail_price_krw"] = prod_df.loc[mismatch_idx, "retail_price_krw"] * \
        np.random.choice([0.8, 0.9, 1.1, 1.2], 30)
    dfs["product_catalog"] = prod_df

    return dfs


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=== Beauty E-commerce Silo Dataset Generator ===\n")

    generators = {
        "product_catalog": ("상품 마스터 (PIM)", gen_products),
        "customer_master": ("고객 마스터 (CRM)", gen_customers),
        "orders": ("주문 (ERP/OMS)", gen_orders),
        "order_items": ("주문상세 (ERP/OMS)", gen_order_items),
        "reviews": ("리뷰 (UGC)", gen_reviews),
        "inventory": ("재고/물류 (WMS)", gen_inventory),
        "promotions": ("프로모션 (Marketing)", gen_promotions),
        "web_events": ("웹/앱 이벤트 (Analytics)", gen_web_events),
        "returns": ("반품/환불 (CS)", gen_returns),
        "suppliers_inbound": ("공급사/입고 (SCM)", gen_suppliers),
    }

    dfs = {}
    total_rows = 0
    total_cols = 0

    for filename, (desc, gen_fn) in generators.items():
        print(f"  Generating {filename} ({desc})...", end=" ", flush=True)
        df = gen_fn()
        dfs[filename] = df
        total_rows += len(df)
        total_cols += len(df.columns)
        print(f"{len(df):,} rows × {len(df.columns)} cols")

    print(f"\n  Injecting cross-file chaos...")
    dfs = inject_chaos(dfs)

    print(f"\n  Writing CSVs to {OUT_DIR}...")
    for filename, df in dfs.items():
        path = os.path.join(OUT_DIR, f"{filename}.csv")
        df.to_csv(path, index=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"    {filename}.csv: {len(df):,} rows × {len(df.columns)} cols ({size_mb:.1f}MB)")

    # Recalculate after chaos
    final_rows = sum(len(df) for df in dfs.values())
    final_cols = sum(len(df.columns) for df in dfs.values())

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(dfs)} files, {final_rows:,} rows × {final_cols:,} cols")
    print(f"{'='*60}")

    print(f"\n=== SILO EFFECTS ===")
    print("고객 ID 매핑:")
    print("  product_catalog : (없음 — 상품 중심)")
    print("  customer_master : customer_id / member_no / user_uid")
    print("  orders          : buyer_code (= member_no)")
    print("  reviews         : user_uid")
    print("  web_events      : user_uid")
    print("  returns         : cust_key (= member_no)")

    print("\n상품 ID 매핑:")
    print("  product_catalog : product_id / sku_code")
    print("  order_items     : item_code (= sku_code) / goods_cd (= product_id)")
    print("  reviews         : goods_cd (= product_id)")
    print("  inventory       : sku (= sku_code)")
    print("  web_events      : prd_no (= product_id)")
    print("  returns         : returned_item_code (= sku_code)")
    print("  suppliers       : sku (= sku_code)")

    print("\n주문 ID 매핑:")
    print("  orders          : order_no / transaction_id")
    print("  order_items     : order_no")
    print("  reviews         : order_ref (= order_no)")
    print("  returns         : ord_num (= order_no)")

    print(f"\n=== CROSS-FILE RELATIONSHIPS (40+) ===")
    rels = [
        "customer→orders (customer_id→buyer_code via member_no)",
        "customer→reviews (customer_id→user_uid)",
        "customer→web_events (customer_id→user_uid)",
        "customer→returns (customer_id→cust_key via member_no)",
        "customer→customer (referrer_id, circular)",
        "orders→order_items (order_no)",
        "orders→promotions (promo_id)",
        "orders→web_events (session_id)",
        "order_items→products (item_code=sku, goods_cd=product_id)",
        "reviews→products (goods_cd=product_id)",
        "reviews→orders (order_ref=order_no)",
        "inventory→products (sku=sku_code)",
        "inventory→warehouses (warehouse_id)",
        "inventory→suppliers (supplier_ref=supplier_id)",
        "products→brands (brand_id)",
        "products→suppliers (supplier_id via brand)",
        "products→ingredients (ingredient_ids, M:N)",
        "products→products (bundle_components, self M:N)",
        "promotions→products (target_products, M:N)",
        "promotions→coupons (coupon_codes, M:N)",
        "returns→orders (ord_num=order_no)",
        "returns→products (returned_item_code=sku)",
        "suppliers→products (sku)",
        "suppliers→brands (brand_id)",
        "suppliers→warehouses (warehouse_dest)",
        "web_events→products (prd_no=product_id)",
        "DERIVED: customer→purchase_history→product→ingredient (4-file join)",
        "DERIVED: return_rate_by_brand = returns→orders→items→products→brand",
        "DERIVED: promo_effectiveness = promotions→orders→order_items (3-file join)",
        "DERIVED: customer_ltv_by_category = customers→orders→items→products (4-file)",
        "DERIVED: supplier_quality_impact = suppliers→inventory→products→reviews (4-file)",
    ]
    for i, r in enumerate(rels, 1):
        print(f"  {i:2d}. {r}")


if __name__ == "__main__":
    main()
