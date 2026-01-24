# Healthcare Silo - FK Relationship Ground Truth

> **Domain**: Healthcare / Hospital Management
> **테이블 수**: 10
> **FK 관계 수**: 14
> **생성일**: 2026-01-22

---

## 테이블 개요

| 테이블 | 레코드 수 | PK | 설명 |
|--------|----------|-----|------|
| departments | 10 | department_id | 진료과/부서 정보 |
| doctors | 25 | doctor_id | 의사 정보 |
| patients | 30 | patient_id | 환자 정보 |
| medications | 20 | medication_id | 약물 정보 |
| appointments | 35 | appointment_id | 진료 예약 |
| diagnoses | 27 | diagnosis_id | 진단 정보 |
| prescriptions | 26 | prescription_id | 처방 정보 |
| medical_records | 20 | record_id | 의료 기록 |
| insurance_claims | 22 | claim_id | 보험 청구 |
| lab_results | 28 | lab_result_id | 검사 결과 |

---

## Ground Truth FK Relationships (14개)

### 1. doctors.dept_code → departments.department_id
- **FK 패턴**: Direct (code suffix)
- **Cardinality**: N:1 (여러 의사가 한 부서 소속)
- **Semantic**: dept는 department의 약어

### 2. patients.primary_doctor_id → doctors.doctor_id
- **FK 패턴**: Direct (entity_id)
- **Cardinality**: N:1 (여러 환자가 한 주치의)
- **Semantic**: primary_doctor는 doctor의 동의어

### 3. appointments.patient_no → patients.patient_id
- **FK 패턴**: Abbreviation (no = number = id)
- **Cardinality**: N:1 (한 환자가 여러 예약)
- **Semantic**: patient_no는 patient 식별자

### 4. appointments.physician_id → doctors.doctor_id
- **FK 패턴**: Synonym (physician = doctor)
- **Cardinality**: N:1 (한 의사가 여러 예약)
- **Semantic**: physician은 doctor의 동의어

### 5. diagnoses.appt_id → appointments.appointment_id
- **FK 패턴**: Abbreviation (appt = appointment)
- **Cardinality**: N:1 (한 예약에서 여러 진단 가능)
- **Semantic**: appt는 appointment의 약어

### 6. diagnoses.pt_id → patients.patient_id
- **FK 패턴**: Abbreviation (pt = patient)
- **Cardinality**: N:1 (한 환자가 여러 진단)
- **Semantic**: pt는 patient의 약어

### 7. diagnoses.diagnosed_by → doctors.doctor_id
- **FK 패턴**: Semantic reference
- **Cardinality**: N:1 (한 의사가 여러 진단)
- **Semantic**: diagnosed_by는 진단한 의사 참조

### 8. prescriptions.diagnosis_ref → diagnoses.diagnosis_id
- **FK 패턴**: Reference suffix (_ref)
- **Cardinality**: N:1 (한 진단에 여러 처방)
- **Semantic**: diagnosis_ref는 diagnosis 참조

### 9. prescriptions.med_code → medications.medication_id
- **FK 패턴**: Abbreviation (med = medication)
- **Cardinality**: N:1 (한 약물이 여러 처방에)
- **Semantic**: med는 medication의 약어

### 10. prescriptions.prescribing_doc → doctors.doctor_id
- **FK 패턴**: Abbreviation + Semantic (doc = doctor)
- **Cardinality**: N:1 (한 의사가 여러 처방)
- **Semantic**: prescribing_doc는 처방 의사

### 11. medical_records.patient_ref → patients.patient_id
- **FK 패턴**: Reference suffix (_ref)
- **Cardinality**: N:1 (한 환자가 여러 기록)
- **Semantic**: patient_ref는 patient 참조

### 12. medical_records.attending_doc → doctors.doctor_id
- **FK 패턴**: Abbreviation (doc = doctor)
- **Cardinality**: N:1 (한 의사가 여러 기록)
- **Semantic**: attending_doc는 담당 의사

### 13. insurance_claims.record_ref → medical_records.record_id
- **FK 패턴**: Reference suffix (_ref)
- **Cardinality**: N:1 (한 기록에 여러 청구 가능)
- **Semantic**: record_ref는 medical_records 참조

### 14. insurance_claims.member_id → patients.patient_id
- **FK 패턴**: Synonym (member = patient in insurance context)
- **Cardinality**: N:1 (한 환자가 여러 청구)
- **Semantic**: member는 보험 가입자 = 환자

### 15. lab_results.order_ref → appointments.appointment_id
- **FK 패턴**: Reference suffix (_ref)
- **Cardinality**: N:1 (한 예약에서 여러 검사)
- **Semantic**: order_ref는 검사 오더 = 예약 참조

### 16. lab_results.patient_identifier → patients.patient_id
- **FK 패턴**: Synonym (identifier = id)
- **Cardinality**: N:1 (한 환자가 여러 검사)
- **Semantic**: patient_identifier는 환자 식별자

### 17. lab_results.ordering_physician → doctors.doctor_id
- **FK 패턴**: Semantic reference (ordering + physician)
- **Cardinality**: N:1 (한 의사가 여러 검사 오더)
- **Semantic**: ordering_physician은 오더한 의사

---

## FK 패턴 분류

### 1. Direct Pattern (직접 명명)
- `doctors.dept_code` → `departments.department_id`
- `patients.primary_doctor_id` → `doctors.doctor_id`

### 2. Abbreviation Pattern (약어)
- `appointments.patient_no` (patient_number의 약어)
- `diagnoses.appt_id` (appointment의 약어)
- `diagnoses.pt_id` (patient의 약어)
- `prescriptions.med_code` (medication의 약어)
- `prescriptions.prescribing_doc` (doctor의 약어)
- `medical_records.attending_doc` (doctor의 약어)

### 3. Synonym Pattern (동의어)
- `appointments.physician_id` (physician = doctor)
- `insurance_claims.member_id` (member = patient in insurance)
- `lab_results.patient_identifier` (identifier = id)

### 4. Reference Suffix Pattern (_ref)
- `prescriptions.diagnosis_ref`
- `medical_records.patient_ref`
- `insurance_claims.record_ref`
- `lab_results.order_ref`

### 5. Semantic Reference (의미적 참조)
- `diagnoses.diagnosed_by` → doctors
- `lab_results.ordering_physician` → doctors

---

## 난이도 평가

| 난이도 | FK 관계 | 설명 |
|--------|---------|------|
| **Easy** | 5 | 직접 명명 패턴 (dept_code, primary_doctor_id 등) |
| **Medium** | 7 | 약어 패턴 (appt_id, pt_id, med_code 등) |
| **Hard** | 5 | 동의어/의미적 참조 (physician_id, member_id, diagnosed_by 등) |

---

## 검증용 쿼리 예시

```sql
-- FK 검증: appointments.patient_no → patients.patient_id
SELECT COUNT(*) as orphans
FROM appointments a
LEFT JOIN patients p ON a.patient_no = p.patient_id
WHERE p.patient_id IS NULL;
-- Expected: 0

-- FK 검증: prescriptions.med_code → medications.medication_id
SELECT COUNT(*) as orphans
FROM prescriptions rx
LEFT JOIN medications m ON rx.med_code = m.medication_id
WHERE m.medication_id IS NULL;
-- Expected: 0
```

---

## Entity Relationship Diagram

```
                    ┌─────────────┐
                    │ departments │
                    │  (10 rows)  │
                    └──────┬──────┘
                           │ 1
                           │
                           │ N
                    ┌──────▼──────┐
    ┌───────────────│   doctors   │───────────────┐
    │               │  (25 rows)  │               │
    │               └──────┬──────┘               │
    │                      │                      │
    │    ┌─────────────────┼─────────────────┐    │
    │    │ N               │ N               │ N  │
    │    │                 │                 │    │
    │    ▼                 ▼                 ▼    ▼
┌───┴────────┐     ┌──────────────┐    ┌─────────┴───────┐
│  patients  │     │ appointments │    │   prescriptions │
│ (30 rows)  │     │  (35 rows)   │    │    (26 rows)    │
└───┬────────┘     └──────┬───────┘    └────────┬────────┘
    │                     │                     │
    │ 1                   │ 1                   │ N
    │                     │                     │
    │ N                   │ N                   │ 1
    ▼                     ▼                     ▼
┌────────────┐     ┌──────────────┐    ┌────────────────┐
│ diagnoses  │     │ lab_results  │    │  medications   │
│ (27 rows)  │     │  (28 rows)   │    │   (20 rows)    │
└────────────┘     └──────────────┘    └────────────────┘
    │
    │ 1
    │
    │ N
    ▼
┌────────────────┐
│medical_records │──────┐
│   (20 rows)    │      │ 1
└────────────────┘      │
                        │ N
                        ▼
               ┌─────────────────┐
               │insurance_claims │
               │   (22 rows)     │
               └─────────────────┘
```

---

## 테스트 기대값

| Metric | Target |
|--------|--------|
| Total FK | 17 |
| Precision | >= 90% |
| Recall | >= 80% |
| F1 Score | >= 85% |

### 난이도별 기대 Recall

| 난이도 | FK 수 | 기대 Recall |
|--------|-------|-------------|
| Easy | 5 | 100% |
| Medium | 7 | 85%+ |
| Hard | 5 | 60%+ |
