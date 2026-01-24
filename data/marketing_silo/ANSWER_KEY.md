# Marketing Data Silo - Analysis Answer Key

이 문서는 `marketing_silo` 데이터셋의 정답지입니다. Ontoloty 시스템이 이 데이터를 분석했을 때 도출해야 하는 올바른 결과를 기술합니다.

---

## 1. 데이터셋 구조

### 1.1 파일 목록 (11개 CSV)
| 파일명 | 레코드 수 | 시스템 출처 | 설명 |
|--------|----------|------------|------|
| customers.csv | 200 | CRM | 고객 마스터 데이터 |
| leads.csv | 150 | CRM | 리드/잠재고객 데이터 |
| products.csv | 50 | E-commerce | 상품 카탈로그 |
| orders.csv | 500 | E-commerce | 주문 데이터 |
| campaigns.csv | 30 | Email Platform | 마케팅 캠페인 |
| email_sends.csv | 2000 | Email Platform | 이메일 발송 |
| email_events.csv | ~580 | Email Platform | 이메일 이벤트 |
| ad_campaigns.csv | 40 | Ad Platform | 광고 캠페인 |
| ad_performance.csv | 800 | Ad Platform | 광고 성과 |
| web_sessions.csv | 1500 | Web Analytics | 웹 세션 |
| conversions.csv | 300 | Attribution | 전환 추적 |

---

## 2. 도메인 감지 (Domain Detection)

### 2.1 Expected Domain
- **Primary Domain**: Marketing / Digital Marketing
- **Confidence**: 0.90+

### 2.2 Domain Indicators
- 마케팅 용어: campaign, lead, conversion, segment, acquisition
- 이커머스 용어: order, product, buyer, cart
- 광고 용어: impressions, clicks, CPC, CPA, CTR
- 이메일 마케팅: email, send, open, click, bounce
- 웹 분석: session, page_views, utm_source, utm_medium

---

## 3. 엔티티 분류 (Entity Classification)

### 3.1 Core Entities

| Entity | Source Table(s) | Primary Key | Classification |
|--------|-----------------|-------------|----------------|
| Customer | customers.csv | customer_id | **MASTER** |
| Lead | leads.csv | lead_id | **TRANSACTIONAL** |
| Product | products.csv | product_id | **MASTER** |
| Order | orders.csv | order_id | **TRANSACTIONAL** |
| Campaign | campaigns.csv | campaign_id | **MASTER** |
| Email Send | email_sends.csv | send_id | **EVENT** |
| Email Event | email_events.csv | event_id | **EVENT** |
| Ad Campaign | ad_campaigns.csv | ad_campaign_id | **MASTER** |
| Ad Performance | ad_performance.csv | perf_id | **EVENT** |
| Web Session | web_sessions.csv | session_id | **EVENT** |
| Conversion | conversions.csv | conv_id | **EVENT** |

### 3.2 Entity Hierarchy
```
Marketing Domain
├── Customer Management
│   ├── Customer (MASTER)
│   └── Lead → Customer (lifecycle)
├── Product Catalog
│   └── Product (MASTER)
├── Transactions
│   └── Order
│       └── Product (M:N)
├── Marketing Campaigns
│   ├── Campaign (Email/SMS/Push)
│   │   ├── Email Send
│   │   │   └── Email Event (Open/Click)
│   │   └── Target Segment
│   └── Ad Campaign (Paid Media)
│       └── Ad Performance (Daily)
├── Web Analytics
│   └── Web Session
│       └── UTM Attribution
└── Attribution
    └── Conversion
        ├── Order
        ├── Ad Campaign
        └── Marketing Campaign
```

---

## 4. 관계 발견 (Relationship Discovery)

### 4.1 명시적 관계 (Explicit - Foreign Key)

| From Table | To Table | Join Keys | Relationship Type | 난이도 |
|------------|----------|-----------|-------------------|--------|
| leads | customers | cust_id = customer_id | N:1 | **Medium** (축약어) |
| orders | customers | buyer_id = customer_id | N:1 | **Hard** (다른 이름) |
| orders | products | prod_code = product_id | N:1 | **Medium** (축약어) |
| email_sends | campaigns | cmp_id = campaign_id | N:1 | **Medium** (축약어) |
| email_sends | customers | cust_no = customer_id | N:1 | **Hard** (축약어 + 다른 패턴) |
| email_events | email_sends | email_send_id = send_id | N:1 | **Easy** (유사 이름) |
| ad_campaigns | campaigns | marketing_campaign_ref = campaign_id | N:1 | **Medium** (다른 이름) |
| ad_performance | ad_campaigns | ad_cmp_id = ad_campaign_id | N:1 | **Medium** (축약어) |
| web_sessions | customers | user_id = customer_id | N:1 | **Hard** (다른 이름) |
| conversions | orders | order_ref = order_id | N:1 | **Medium** (다른 이름) |
| conversions | ad_campaigns | attributed_ad_id = ad_campaign_id | N:1 | **Easy** |
| conversions | campaigns | attributed_cmp_id = campaign_id | N:1 | **Medium** (축약어) |

### 4.2 데이터 사일로 불일치 (Key Findings)

1. **고객 ID 불일치** (가장 심각)
   - `customers.customer_id` (정식)
   - `leads.cust_id` (축약)
   - `orders.buyer_id` (완전히 다른 이름!)
   - `email_sends.cust_no` (축약 + 다른 패턴)
   - `web_sessions.user_id` (다른 이름)

2. **캠페인 ID 불일치**
   - `campaigns.campaign_id` (정식)
   - `email_sends.cmp_id` (축약)
   - `ad_campaigns.marketing_campaign_ref` (다른 이름)
   - `conversions.attributed_cmp_id` (축약)

3. **상품 ID 불일치**
   - `products.product_id` (정식)
   - `orders.prod_code` (축약)

4. **광고 캠페인 ID 불일치**
   - `ad_campaigns.ad_campaign_id` (정식)
   - `ad_performance.ad_cmp_id` (축약)

---

## 5. 축약어 매핑 (Semantic Abbreviations)

시스템이 감지해야 하는 축약어 패턴:

| 축약어 | 원형 | 사용 위치 |
|--------|------|----------|
| cust | customer | leads.cust_id, email_sends.cust_no |
| cmp | campaign | email_sends.cmp_id, ad_performance.ad_cmp_id, conversions.attributed_cmp_id |
| prod | product | orders.prod_code |
| conv | conversion | conversions.conv_id |
| perf | performance | ad_performance.perf_id |

---

## 6. 데이터 품질 이슈

### 6.1 Missing Values
| Table | Column | Missing Count | Impact |
|-------|--------|---------------|--------|
| leads | cust_id | ~105 (70%) | 미전환 리드 |
| orders | - | 0 | Clean |
| web_sessions | user_id | ~450 (30%) | 비로그인 사용자 |
| ad_campaigns | marketing_campaign_ref | ~20 (50%) | 독립 광고 캠페인 |
| conversions | order_ref | ~60 (20%) | 비구매 전환 |
| conversions | attributed_ad_id | ~180 (60%) | 광고 외 전환 |
| conversions | attributed_cmp_id | ~210 (70%) | 캠페인 외 전환 |

### 6.2 Data Quality Issues to Detect

1. **Lead-Customer 연결**: 전환된 리드(30%)만 cust_id 보유
2. **비로그인 세션**: web_sessions의 30%가 user_id 없음
3. **Multi-Attribution Gap**: conversions에서 광고/캠페인 둘 다 없는 케이스 존재

---

## 7. 통계적 인사이트 (Expected Insights)

### 7.1 Customer Statistics
- **Total Customers**: 200
- **By Segment**:
  - VIP: ~10 (5%)
  - Premium: ~30 (15%)
  - Standard: ~80 (40%)
  - Basic: ~60 (30%)
  - Inactive: ~20 (10%)

### 7.2 Order Statistics
- **Total Orders**: 500
- **By Status**:
  - COMPLETED: ~350 (70%)
  - SHIPPED: ~75 (15%)
  - PROCESSING: ~25 (5%)
  - CANCELLED: ~25 (5%)
  - REFUNDED: ~25 (5%)

### 7.3 Campaign Performance
- **Total Campaigns**: 30
- **By Type**:
  - PROMOTIONAL, NEWSLETTER, RETENTION, WINBACK, WELCOME, SEASONAL

### 7.4 Email Performance (예상)
- **Total Sends**: 2000
- **Delivery Rate**: ~92%
- **Open Rate**: ~25% (of delivered)
- **Click Rate**: ~7.5% (of delivered), ~30% (of opened)

### 7.5 Ad Performance (예상)
- **Total Ad Campaigns**: 40
- **By Platform**: GOOGLE, FACEBOOK, INSTAGRAM, NAVER, KAKAO
- **Avg CTR**: 1-5%
- **Avg Conversion Rate**: 1-10% of clicks

---

## 8. 온톨로지 구축 정답

### 8.1 Classes (Expected)
```
Thing
├── Organization
│   └── Company (implied)
├── Person
│   ├── Customer
│   │   └── Segment (VIP, Premium, Standard, Basic)
│   └── Lead
│       └── Lead Status (NEW → CONVERTED)
├── Product
│   └── Category
├── Event
│   ├── Transaction
│   │   └── Order
│   ├── MarketingEvent
│   │   ├── Campaign
│   │   │   ├── EmailCampaign
│   │   │   └── AdCampaign
│   │   ├── EmailSend
│   │   └── EmailEvent (Open, Click)
│   ├── WebEvent
│   │   └── Session
│   └── Conversion
└── Metric
    └── AdPerformance
```

### 8.2 Object Properties (Expected)
```
belongsToCustomer: Lead → Customer (converted)
placedBy: Order → Customer (buyer_id = customer_id)
contains: Order → Product
targetedBy: Customer → Campaign
receivedEmail: Customer → EmailSend
triggeredEvent: EmailSend → EmailEvent
linkedToCampaign: AdCampaign → Campaign
measuredBy: AdCampaign → AdPerformance
visitedBy: WebSession → Customer
attributedTo: Conversion → {Order, AdCampaign, Campaign}
```

---

## 9. 검증 체크리스트

Ontoloty 시스템이 이 데이터를 분석할 때 다음을 검증:

### 9.1 Domain Detection
- [ ] Marketing/Digital Marketing 도메인 감지
- [ ] Confidence ≥ 0.85

### 9.2 Entity Discovery
- [ ] 최소 8개 주요 엔티티 식별
- [ ] Master/Transactional/Event 분류 정확

### 9.3 Relationship Discovery (핵심!)
**기대하는 FK 관계 12개:**
- [ ] `leads.cust_id → customers.customer_id` (축약어)
- [ ] `orders.buyer_id → customers.customer_id` (다른 이름)
- [ ] `orders.prod_code → products.product_id` (축약어)
- [ ] `email_sends.cmp_id → campaigns.campaign_id` (축약어)
- [ ] `email_sends.cust_no → customers.customer_id` (축약어 + 패턴)
- [ ] `email_events.email_send_id → email_sends.send_id` (유사 이름)
- [ ] `ad_campaigns.marketing_campaign_ref → campaigns.campaign_id` (다른 이름)
- [ ] `ad_performance.ad_cmp_id → ad_campaigns.ad_campaign_id` (축약어)
- [ ] `web_sessions.user_id → customers.customer_id` (다른 이름)
- [ ] `conversions.order_ref → orders.order_id` (다른 이름)
- [ ] `conversions.attributed_ad_id → ad_campaigns.ad_campaign_id` (유사 이름)
- [ ] `conversions.attributed_cmp_id → campaigns.campaign_id` (축약어)

### 9.4 Semantic Matching
- [ ] cust → customer 매핑 감지
- [ ] cmp → campaign 매핑 감지
- [ ] prod → product 매핑 감지
- [ ] buyer → customer 의미 연결
- [ ] user → customer 의미 연결

---

## 10. 난이도 평가

| 항목 | 난이도 | 설명 |
|------|--------|------|
| Domain Detection | ★☆☆ | 명확한 마케팅 도메인 |
| Entity Classification | ★★☆ | 다양한 이벤트 타입 |
| Explicit Relationships | ★★☆ | FK 존재하나 네이밍 불일치 |
| Semantic Matching | ★★★ | 축약어, 다른 이름 많음 |
| Data Quality Issues | ★★☆ | 의도적 누락 데이터 |
| Cross-System Integration | ★★★ | 5개 시스템 통합 필요 |

**Overall Difficulty**: ★★★ (Hard)

> airport_silo 대비 더 어려움: 축약어와 완전히 다른 이름(buyer_id, user_id)이 더 많음

---

## 11. 추가 참고 사항

### 11.1 데이터 생성 로직
- 2023-06-01 ~ 2023-12-01 기간 데이터
- 한국 마케팅 시나리오 (한글 이름, 원화 금액)
- 실제 마케팅 플랫폼 용어 사용

### 11.2 의도적 데이터 사일로 패턴
1. **CRM 시스템**: customers, leads (customer_id, cust_id)
2. **E-commerce 시스템**: orders, products (buyer_id, prod_code)
3. **Email Marketing 플랫폼**: campaigns, email_sends, email_events (cmp_id, cust_no)
4. **Ad Platform**: ad_campaigns, ad_performance (ad_campaign_id, ad_cmp_id)
5. **Web Analytics**: web_sessions (user_id)
6. **Attribution System**: conversions (order_ref, attributed_*)

### 11.3 핵심 도전 과제
1. **Customer 엔티티 통합**: 5가지 다른 ID로 분산된 고객 데이터
2. **Campaign 연결**: 마케팅 캠페인 ↔ 광고 캠페인 ↔ 전환 연결
3. **Attribution 분석**: 멀티터치 어트리뷰션 경로 추적

---

*Generated by: Claude (Data Set Creator)*
*Date: 2026-01-20*
*Version: 1.0*
