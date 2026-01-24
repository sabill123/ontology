"""
Marketing Silo Dataset Generator

마케팅 부서에서 사용하는 데이터 사일로 시뮬레이션:
- CRM 시스템: customers, leads
- Email Marketing Platform: campaigns, email_sends, email_events
- Ad Platform: ad_campaigns, ad_performance
- E-commerce: orders, products
- Web Analytics: web_sessions

의도적 데이터 사일로 패턴:
1. 동일 엔티티의 다른 식별자 (customer_id vs cust_id vs buyer_id)
2. 축약어 사용 (cust, prod, conv, cmp 등)
3. 코드 체계 불일치 (채널 코드, 캠페인 타입)
4. 명시적 FK 부재
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

OUTPUT_DIR = "/Users/jaeseokhan/Desktop/Work/ontoloty/data/marketing_silo"

# ============================================
# 1. CUSTOMERS (CRM System)
# ============================================
def generate_customers(n=200):
    """고객 마스터 데이터"""
    first_names = ["김", "이", "박", "최", "정", "강", "조", "윤", "장", "임"]
    last_names = ["민준", "서연", "하준", "지우", "도윤", "서윤", "예준", "지민", "시우", "수아",
                  "지호", "하은", "준서", "지유", "현우", "채원", "지훈", "수빈", "건우", "지아"]

    segments = ["VIP", "Premium", "Standard", "Basic", "Inactive"]
    segment_weights = [0.05, 0.15, 0.40, 0.30, 0.10]

    channels = ["ONLINE", "OFFLINE", "MOBILE", "PARTNER"]
    channel_weights = [0.45, 0.25, 0.25, 0.05]

    regions = ["서울", "경기", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "강원"]

    customers = []
    for i in range(n):
        customer_id = f"CUST{str(i+1).zfill(6)}"
        name = f"{random.choice(first_names)}{random.choice(last_names)}"
        email = f"user{i+1}@{'gmail.com' if random.random() > 0.3 else 'naver.com'}"

        signup_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        segment = random.choices(segments, weights=segment_weights)[0]
        acquisition_channel = random.choices(channels, weights=channel_weights)[0]
        region = random.choice(regions)

        lifetime_value = round(random.gauss(500000, 200000) if segment == "VIP" else
                              random.gauss(200000, 80000) if segment == "Premium" else
                              random.gauss(80000, 30000) if segment == "Standard" else
                              random.gauss(30000, 15000), 0)
        lifetime_value = max(0, lifetime_value)

        customers.append({
            "customer_id": customer_id,
            "customer_name": name,
            "email": email,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "segment": segment,
            "acquisition_channel": acquisition_channel,
            "region": region,
            "lifetime_value": int(lifetime_value),
            "is_active": 1 if segment != "Inactive" else 0,
            "last_purchase_date": (signup_date + timedelta(days=random.randint(1, 300))).strftime("%Y-%m-%d") if segment != "Inactive" else None
        })

    return pd.DataFrame(customers)


# ============================================
# 2. LEADS (CRM System - 다른 네이밍)
# ============================================
def generate_leads(n=150):
    """리드/잠재고객 데이터 (다른 네이밍 체계)"""
    sources = ["GOOGLE_ADS", "FACEBOOK", "ORGANIC", "REFERRAL", "WEBINAR", "CONTENT"]
    source_weights = [0.30, 0.25, 0.20, 0.10, 0.08, 0.07]

    statuses = ["NEW", "CONTACTED", "QUALIFIED", "PROPOSAL", "CONVERTED", "LOST"]
    status_weights = [0.15, 0.20, 0.25, 0.15, 0.15, 0.10]

    leads = []
    for i in range(n):
        lead_id = f"LD{str(i+1).zfill(5)}"

        # 일부 리드는 기존 고객으로 전환됨 (cust_id 사용 - 축약어!)
        converted_cust = f"CUST{str(random.randint(1, 100)).zfill(6)}" if random.random() < 0.3 else None

        source = random.choices(sources, weights=source_weights)[0]
        status = random.choices(statuses, weights=status_weights)[0]

        if converted_cust:
            status = "CONVERTED"

        created_at = datetime(2023, 6, 1) + timedelta(days=random.randint(0, 200))

        score = random.randint(10, 100)
        if status == "QUALIFIED":
            score = random.randint(60, 100)
        elif status == "LOST":
            score = random.randint(10, 40)

        leads.append({
            "lead_id": lead_id,
            "lead_email": f"prospect{i+1}@company{random.randint(1,50)}.com",
            "source": source,
            "status": status,
            "lead_score": score,
            "created_at": created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "cust_id": converted_cust,  # 축약어! customer_id 아님
            "assigned_rep": f"REP{str(random.randint(1, 10)).zfill(3)}"
        })

    return pd.DataFrame(leads)


# ============================================
# 3. PRODUCTS (E-commerce)
# ============================================
def generate_products(n=50):
    """상품 카탈로그"""
    categories = ["전자기기", "패션", "뷰티", "식품", "가구", "스포츠"]

    products = []
    for i in range(n):
        product_id = f"PROD{str(i+1).zfill(4)}"
        category = random.choice(categories)

        base_price = random.randint(10000, 500000)

        products.append({
            "product_id": product_id,
            "product_name": f"{category} 상품 {i+1}",
            "category": category,
            "price": base_price,
            "cost": int(base_price * random.uniform(0.4, 0.7)),
            "margin_rate": round(random.uniform(0.2, 0.5), 2),
            "stock_qty": random.randint(0, 500),
            "is_active": 1 if random.random() > 0.1 else 0
        })

    return pd.DataFrame(products)


# ============================================
# 4. ORDERS (E-commerce - 다른 키 네이밍)
# ============================================
def generate_orders(customers_df, products_df, n=500):
    """주문 데이터 (buyer_id 사용 - customer_id 아님!)"""
    orders = []

    active_customers = customers_df[customers_df["is_active"] == 1]["customer_id"].tolist()
    product_ids = products_df["product_id"].tolist()

    for i in range(n):
        order_id = f"ORD{str(i+1).zfill(6)}"
        buyer_id = random.choice(active_customers)  # buyer_id 사용!
        prod_code = random.choice(product_ids)  # prod_code 사용!

        order_date = datetime(2023, 6, 1) + timedelta(days=random.randint(0, 200))

        quantity = random.randint(1, 5)
        product = products_df[products_df["product_id"] == prod_code].iloc[0]
        unit_price = product["price"]

        # 할인
        discount_rate = random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20])
        total_amount = int(unit_price * quantity * (1 - discount_rate))

        status = random.choices(
            ["COMPLETED", "SHIPPED", "PROCESSING", "CANCELLED", "REFUNDED"],
            weights=[0.70, 0.15, 0.05, 0.05, 0.05]
        )[0]

        orders.append({
            "order_id": order_id,
            "buyer_id": buyer_id,  # customer_id가 아닌 buyer_id!
            "prod_code": prod_code,  # product_id가 아닌 prod_code!
            "order_date": order_date.strftime("%Y-%m-%d"),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_rate": discount_rate,
            "total_amount": total_amount,
            "status": status,
            "channel": random.choice(["WEB", "APP", "CALL"])
        })

    return pd.DataFrame(orders)


# ============================================
# 5. CAMPAIGNS (Email Marketing Platform)
# ============================================
def generate_campaigns(n=30):
    """마케팅 캠페인 데이터"""
    campaign_types = ["PROMOTIONAL", "NEWSLETTER", "RETENTION", "WINBACK", "WELCOME", "SEASONAL"]
    channels = ["EMAIL", "SMS", "PUSH", "KAKAO"]

    campaigns = []
    for i in range(n):
        campaign_id = f"CMP{str(i+1).zfill(4)}"

        start_date = datetime(2023, 6, 1) + timedelta(days=random.randint(0, 180))
        end_date = start_date + timedelta(days=random.randint(1, 30))

        campaign_type = random.choice(campaign_types)
        channel = random.choice(channels)

        budget = random.randint(100000, 5000000)

        campaigns.append({
            "campaign_id": campaign_id,
            "campaign_name": f"{campaign_type} 캠페인 {i+1}",
            "campaign_type": campaign_type,
            "channel": channel,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "budget": budget,
            "target_segment": random.choice(["VIP", "Premium", "Standard", "Basic", "ALL"]),
            "status": "COMPLETED" if end_date < datetime(2023, 12, 1) else "ACTIVE"
        })

    return pd.DataFrame(campaigns)


# ============================================
# 6. EMAIL_SENDS (Email Marketing - 축약어 사용)
# ============================================
def generate_email_sends(campaigns_df, customers_df, n=2000):
    """이메일 발송 데이터 (축약어: cmp_id, cust_no)"""
    email_campaigns = campaigns_df[campaigns_df["channel"] == "EMAIL"]["campaign_id"].tolist()
    customer_ids = customers_df["customer_id"].tolist()

    sends = []
    for i in range(n):
        send_id = f"ES{str(i+1).zfill(6)}"
        cmp_id = random.choice(email_campaigns) if email_campaigns else f"CMP{str(random.randint(1, 10)).zfill(4)}"
        cust_no = random.choice(customer_ids)  # cust_no 사용!

        send_time = datetime(2023, 6, 1) + timedelta(days=random.randint(0, 180),
                                                      hours=random.randint(8, 20))

        # 발송 상태
        status = random.choices(
            ["DELIVERED", "BOUNCED", "FAILED"],
            weights=[0.92, 0.05, 0.03]
        )[0]

        sends.append({
            "send_id": send_id,
            "cmp_id": cmp_id,  # campaign_id 축약!
            "cust_no": cust_no,  # customer_id 축약!
            "send_time": send_time.strftime("%Y-%m-%d %H:%M:%S"),
            "email_subject": f"[프로모션] 특별 할인 안내 #{i+1}",
            "status": status
        })

    return pd.DataFrame(sends)


# ============================================
# 7. EMAIL_EVENTS (Email Marketing - 이벤트 추적)
# ============================================
def generate_email_events(email_sends_df, n=3000):
    """이메일 이벤트 (오픈, 클릭)"""
    delivered_sends = email_sends_df[email_sends_df["status"] == "DELIVERED"]["send_id"].tolist()

    events = []
    event_id = 0

    for send_id in delivered_sends:
        # 오픈 여부 (약 25%)
        if random.random() < 0.25:
            event_id += 1
            send_row = email_sends_df[email_sends_df["send_id"] == send_id].iloc[0]
            send_time = datetime.strptime(send_row["send_time"], "%Y-%m-%d %H:%M:%S")

            open_time = send_time + timedelta(hours=random.randint(0, 48))

            events.append({
                "event_id": f"EVT{str(event_id).zfill(6)}",
                "email_send_id": send_id,  # send_id 참조
                "event_type": "OPEN",
                "event_time": open_time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": random.choice(["MOBILE", "DESKTOP", "TABLET"]),
                "link_url": None
            })

            # 클릭 여부 (오픈한 사람 중 30%)
            if random.random() < 0.30:
                event_id += 1
                click_time = open_time + timedelta(minutes=random.randint(1, 30))

                events.append({
                    "event_id": f"EVT{str(event_id).zfill(6)}",
                    "email_send_id": send_id,
                    "event_type": "CLICK",
                    "event_time": click_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "device": random.choice(["MOBILE", "DESKTOP", "TABLET"]),
                    "link_url": f"https://shop.example.com/promo/{random.randint(1, 100)}"
                })

    return pd.DataFrame(events)


# ============================================
# 8. AD_CAMPAIGNS (Ad Platform - 다른 코드 체계)
# ============================================
def generate_ad_campaigns(n=40):
    """광고 캠페인 (광고 플랫폼 - 다른 ID 체계)"""
    platforms = ["GOOGLE", "FACEBOOK", "INSTAGRAM", "NAVER", "KAKAO"]
    objectives = ["AWARENESS", "TRAFFIC", "CONVERSION", "LEAD_GEN"]

    ad_campaigns = []
    for i in range(n):
        # AD 플랫폼은 다른 ID 체계 사용!
        ad_campaign_id = f"AD{str(i+1).zfill(5)}"

        # 일부는 마케팅 캠페인과 연결 (다른 ID 형식으로!)
        linked_campaign = f"CMP{str(random.randint(1, 30)).zfill(4)}" if random.random() < 0.5 else None

        platform = random.choice(platforms)
        objective = random.choice(objectives)

        start_date = datetime(2023, 6, 1) + timedelta(days=random.randint(0, 150))

        daily_budget = random.randint(10000, 500000)

        ad_campaigns.append({
            "ad_campaign_id": ad_campaign_id,
            "ad_name": f"{platform} {objective} 캠페인",
            "platform": platform,
            "objective": objective,
            "marketing_campaign_ref": linked_campaign,  # FK인데 다른 이름!
            "daily_budget": daily_budget,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "status": random.choice(["ACTIVE", "PAUSED", "COMPLETED"])
        })

    return pd.DataFrame(ad_campaigns)


# ============================================
# 9. AD_PERFORMANCE (Ad Platform - 일별 성과)
# ============================================
def generate_ad_performance(ad_campaigns_df, n=800):
    """광고 성과 데이터"""
    ad_campaign_ids = ad_campaigns_df["ad_campaign_id"].tolist()

    performance = []
    for i in range(n):
        perf_id = f"PERF{str(i+1).zfill(6)}"
        ad_cmp_id = random.choice(ad_campaign_ids)  # ad_cmp_id 축약!

        date = datetime(2023, 6, 1) + timedelta(days=random.randint(0, 180))

        impressions = random.randint(1000, 100000)
        clicks = int(impressions * random.uniform(0.01, 0.05))
        conversions = int(clicks * random.uniform(0.01, 0.10))
        spend = random.randint(5000, 200000)

        performance.append({
            "perf_id": perf_id,
            "ad_cmp_id": ad_cmp_id,  # ad_campaign_id 축약!
            "date": date.strftime("%Y-%m-%d"),
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "spend": spend,
            "cpc": round(spend / clicks, 2) if clicks > 0 else 0,
            "cpa": round(spend / conversions, 2) if conversions > 0 else 0,
            "ctr": round(clicks / impressions * 100, 2) if impressions > 0 else 0
        })

    return pd.DataFrame(performance)


# ============================================
# 10. WEB_SESSIONS (Web Analytics - 또 다른 ID)
# ============================================
def generate_web_sessions(customers_df, n=1500):
    """웹 세션 데이터 (또 다른 고객 ID 체계)"""
    customer_ids = customers_df["customer_id"].tolist()

    sessions = []
    for i in range(n):
        session_id = f"SESS{str(i+1).zfill(8)}"

        # 로그인한 사용자 (70%)
        if random.random() < 0.70:
            user_id = random.choice(customer_ids)  # user_id 사용!
        else:
            user_id = None  # 비로그인

        start_time = datetime(2023, 6, 1) + timedelta(
            days=random.randint(0, 180),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )

        duration_sec = random.randint(10, 1800)
        page_views = random.randint(1, 20)

        # UTM 파라미터
        utm_sources = ["google", "facebook", "naver", "direct", "email", "referral"]
        utm_mediums = ["cpc", "organic", "social", "email", "referral", "(none)"]

        sessions.append({
            "session_id": session_id,
            "user_id": user_id,  # customer_id가 아닌 user_id!
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": duration_sec,
            "page_views": page_views,
            "utm_source": random.choice(utm_sources),
            "utm_medium": random.choice(utm_mediums),
            "utm_campaign": f"cmp_{random.randint(1, 30)}" if random.random() < 0.4 else None,
            "device": random.choice(["mobile", "desktop", "tablet"]),
            "converted": 1 if random.random() < 0.05 else 0
        })

    return pd.DataFrame(sessions)


# ============================================
# 11. CONVERSIONS (Attribution - 전환 추적)
# ============================================
def generate_conversions(orders_df, ad_campaigns_df, campaigns_df, n=300):
    """전환 데이터 (여러 시스템 연결)"""
    order_ids = orders_df[orders_df["status"] == "COMPLETED"]["order_id"].tolist()
    ad_ids = ad_campaigns_df["ad_campaign_id"].tolist()
    campaign_ids = campaigns_df["campaign_id"].tolist()

    conversions = []
    for i in range(n):
        conv_id = f"CONV{str(i+1).zfill(6)}"

        # 주문과 연결
        order_ref = random.choice(order_ids) if random.random() < 0.8 else None

        # 어트리뷰션 채널
        if random.random() < 0.4:
            # 광고 전환
            attributed_ad = random.choice(ad_ids)
            attributed_campaign = None
        elif random.random() < 0.6:
            # 마케팅 캠페인 전환
            attributed_ad = None
            attributed_campaign = random.choice(campaign_ids)
        else:
            # 직접 전환
            attributed_ad = None
            attributed_campaign = None

        conv_time = datetime(2023, 6, 1) + timedelta(days=random.randint(0, 180))

        conversions.append({
            "conv_id": conv_id,
            "order_ref": order_ref,  # order_id가 아닌 order_ref!
            "attributed_ad_id": attributed_ad,
            "attributed_cmp_id": attributed_campaign,  # campaign_id 축약!
            "conv_time": conv_time.strftime("%Y-%m-%d %H:%M:%S"),
            "conv_value": random.randint(10000, 500000),
            "conv_type": random.choice(["PURCHASE", "SIGNUP", "LEAD"])
        })

    return pd.DataFrame(conversions)


# ============================================
# MAIN
# ============================================
def main():
    print("Generating marketing silo dataset...")

    # 생성
    customers = generate_customers(200)
    leads = generate_leads(150)
    products = generate_products(50)
    orders = generate_orders(customers, products, 500)
    campaigns = generate_campaigns(30)
    email_sends = generate_email_sends(campaigns, customers, 2000)
    email_events = generate_email_events(email_sends, 3000)
    ad_campaigns = generate_ad_campaigns(40)
    ad_performance = generate_ad_performance(ad_campaigns, 800)
    web_sessions = generate_web_sessions(customers, 1500)
    conversions = generate_conversions(orders, ad_campaigns, campaigns, 300)

    # 저장
    customers.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
    leads.to_csv(f"{OUTPUT_DIR}/leads.csv", index=False)
    products.to_csv(f"{OUTPUT_DIR}/products.csv", index=False)
    orders.to_csv(f"{OUTPUT_DIR}/orders.csv", index=False)
    campaigns.to_csv(f"{OUTPUT_DIR}/campaigns.csv", index=False)
    email_sends.to_csv(f"{OUTPUT_DIR}/email_sends.csv", index=False)
    email_events.to_csv(f"{OUTPUT_DIR}/email_events.csv", index=False)
    ad_campaigns.to_csv(f"{OUTPUT_DIR}/ad_campaigns.csv", index=False)
    ad_performance.to_csv(f"{OUTPUT_DIR}/ad_performance.csv", index=False)
    web_sessions.to_csv(f"{OUTPUT_DIR}/web_sessions.csv", index=False)
    conversions.to_csv(f"{OUTPUT_DIR}/conversions.csv", index=False)

    print(f"\nGenerated files:")
    print(f"  - customers.csv: {len(customers)} rows")
    print(f"  - leads.csv: {len(leads)} rows")
    print(f"  - products.csv: {len(products)} rows")
    print(f"  - orders.csv: {len(orders)} rows")
    print(f"  - campaigns.csv: {len(campaigns)} rows")
    print(f"  - email_sends.csv: {len(email_sends)} rows")
    print(f"  - email_events.csv: {len(email_events)} rows")
    print(f"  - ad_campaigns.csv: {len(ad_campaigns)} rows")
    print(f"  - ad_performance.csv: {len(ad_performance)} rows")
    print(f"  - web_sessions.csv: {len(web_sessions)} rows")
    print(f"  - conversions.csv: {len(conversions)} rows")

    print("\nDone!")


if __name__ == "__main__":
    main()
