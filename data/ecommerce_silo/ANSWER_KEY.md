# E-commerce Dataset - FK Relationship Answer Key

## Dataset Overview
- **Domain**: E-commerce / Online Retail
- **Tables**: 10개
- **Total FK Relationships**: 20개

## Table Summary

| Table | Primary Key | Rows | Description |
|-------|-------------|------|-------------|
| customers | cust_id | 15 | 고객 정보 |
| products | prod_code | 20 | 상품 정보 |
| categories | category_id | 12 | 상품 카테고리 |
| vendors | vendor_id | 13 | 판매자/입점업체 |
| orders | order_id | 25 | 주문 정보 |
| order_items | item_id | 30 | 주문 상세 항목 |
| payments | payment_id | 25 | 결제 정보 |
| shipping | shipping_id | 25 | 배송 정보 |
| reviews | review_id | 23 | 상품 리뷰 |
| promotions | promo_id | 10 | 프로모션/할인 |
| order_promotions | id | 16 | 주문-프로모션 연결 |

## Ground Truth FK Relationships (20개)

### 1. Direct FK (명시적 ID 참조) - 4개

| # | Source Table | Source Column | Target Table | Target Column | Pattern |
|---|--------------|---------------|--------------|---------------|---------|
| 1 | order_promotions | order_id | orders | order_id | Direct match |
| 2 | categories | parent_cat | categories | category_id | Self-reference (abbreviation) |
| 3 | promotions | applicable_category | categories | category_id | Semantic reference |
| 4 | orders | shipping_address | - | - | NOT FK (text field) |

### 2. Abbreviation FK (축약형) - 6개

| # | Source Table | Source Column | Target Table | Target Column | Pattern |
|---|--------------|---------------|--------------|---------------|---------|
| 5 | products | cat_id | categories | category_id | cat → category |
| 6 | order_items | ord_ref | orders | order_id | ord → order, ref → id |
| 7 | order_items | item_code | products | prod_code | item → prod, code 일치 |
| 8 | payments | order_ref | orders | order_id | ref → id |
| 9 | shipping | order_no | orders | order_id | no → id |
| 10 | order_promotions | applied_promo | promotions | promo_id | promo 축약 |

### 3. Synonym FK (동의어/유사어) - 6개

| # | Source Table | Source Column | Target Table | Target Column | Pattern |
|---|--------------|---------------|--------------|---------------|---------|
| 11 | orders | buyer_id | customers | cust_id | buyer = customer (동의어) |
| 12 | products | seller_id | vendors | vendor_id | seller = vendor (동의어) |
| 13 | reviews | reviewed_product | products | prod_code | product 의미 참조 |
| 14 | reviews | reviewed_by | customers | cust_id | by → 사람 = customer |
| 15 | shipping | shipped_by | vendors | vendor_id | by → 배송주체 = vendor |
| 16 | categories | parent_cat | categories | category_id | Self-reference |

### 4. Semantic FK (의미적 추론 필요) - 4개

| # | Source Table | Source Column | Target Table | Target Column | Pattern |
|---|--------------|---------------|--------------|---------------|---------|
| 17 | promotions | applicable_category | categories | category_id | 적용 카테고리 |
| 18 | order_promotions | applied_promo | promotions | promo_id | 적용된 프로모션 |
| 19 | reviews | reviewed_product | products | prod_code | 리뷰 대상 상품 |
| 20 | reviews | reviewed_by | customers | cust_id | 리뷰 작성자 |

## Deduplicated FK List (중복 제거 후 최종 20개)

| # | Source | Target | Difficulty |
|---|--------|--------|------------|
| 1 | orders.buyer_id | customers.cust_id | Medium (synonym) |
| 2 | products.cat_id | categories.category_id | Easy (abbreviation) |
| 3 | products.seller_id | vendors.vendor_id | Medium (synonym) |
| 4 | categories.parent_cat | categories.category_id | Medium (self-ref + abbr) |
| 5 | order_items.ord_ref | orders.order_id | Easy (abbreviation) |
| 6 | order_items.item_code | products.prod_code | Medium (semantic) |
| 7 | payments.order_ref | orders.order_id | Easy (abbreviation) |
| 8 | shipping.order_no | orders.order_id | Easy (abbreviation) |
| 9 | shipping.shipped_by | vendors.vendor_id | Hard (semantic) |
| 10 | reviews.reviewed_product | products.prod_code | Hard (semantic) |
| 11 | reviews.reviewed_by | customers.cust_id | Hard (semantic) |
| 12 | promotions.applicable_category | categories.category_id | Hard (semantic) |
| 13 | order_promotions.order_id | orders.order_id | Easy (direct) |
| 14 | order_promotions.applied_promo | promotions.promo_id | Medium (abbreviation) |

## FK Pattern Distribution

### By Difficulty
- **Easy (Direct/Simple Abbreviation)**: 5개 (35.7%)
- **Medium (Synonym/Complex Abbreviation)**: 5개 (35.7%)
- **Hard (Semantic Inference Required)**: 4개 (28.6%)

### By Pattern Type
- **Direct Match**: 1개
- **Abbreviation**: 6개 (cat_id, ord_ref, order_ref, order_no, parent_cat, applied_promo)
- **Synonym/Alias**: 3개 (buyer_id, seller_id, item_code)
- **Semantic Reference**: 4개 (reviewed_product, reviewed_by, shipped_by, applicable_category)

## Expected Detection Results

### Rule-Based Detection (예상)
다음 FK는 규칙 기반으로 탐지 가능:
1. ✅ order_promotions.order_id → orders.order_id (exact match)
2. ✅ products.cat_id → categories.category_id (suffix match + data)
3. ✅ order_items.ord_ref → orders.order_id (data pattern)
4. ✅ payments.order_ref → orders.order_id (suffix match + data)
5. ✅ shipping.order_no → orders.order_id (data pattern)
6. ✅ order_promotions.applied_promo → promotions.promo_id (data pattern)
7. ✅ categories.parent_cat → categories.category_id (data pattern)
8. ✅ products.seller_id → vendors.vendor_id (suffix match + data)

**예상 Rule-Based 탐지**: 8/14 (57.1%)

### LLM Enhancement Required (예상)
다음 FK는 LLM 의미 분석이 필요:
1. 🤖 orders.buyer_id → customers.cust_id (buyer = customer synonym)
2. 🤖 order_items.item_code → products.prod_code (item = product synonym)
3. 🤖 reviews.reviewed_product → products.prod_code (semantic)
4. 🤖 reviews.reviewed_by → customers.cust_id (semantic)
5. 🤖 shipping.shipped_by → vendors.vendor_id (semantic)
6. 🤖 promotions.applicable_category → categories.category_id (semantic)

**예상 LLM 추가 탐지**: 6개

## Validation Criteria

### Perfect Score Conditions
- **Precision**: 탐지된 FK 중 정답 비율
- **Recall**: 전체 정답 중 탐지 비율
- **F1 Score**: Precision과 Recall의 조화평균

### Target Metrics
- Rule-Based Only: Recall ≥ 57% (8/14)
- With LLM Enhancement: Recall ≥ 85% (12/14)
- Combined (Full Pipeline): Recall = 100% (14/14)

## Notes

### Design Principles
1. **다양한 FK 패턴**: Healthcare 대비 더 많은 synonym/semantic 패턴 포함
2. **현실적 데이터**: 실제 이커머스 도메인의 테이블 구조 반영
3. **난이도 분포**: Easy 35%, Medium 35%, Hard 30%로 균형있는 분포
4. **Self-Reference**: categories.parent_cat 포함 (계층 구조)
5. **다대다 관계**: order_promotions 브릿지 테이블 포함

### Challenging Cases
- `buyer_id` vs `cust_id`: 동의어 관계 (구매자 = 고객)
- `seller_id` vs `vendor_id`: 동의어 관계 (판매자 = 입점업체)
- `item_code` vs `prod_code`: 의미적 관계 (아이템 = 상품)
- `reviewed_by`: 문맥상 고객을 지칭
- `shipped_by`: 문맥상 판매자/물류업체를 지칭
