# HR/인사관리 Dataset - FK Relationship Answer Key

## Dataset Overview
- **Domain**: HR / Human Resources (인사관리)
- **Tables**: 11개
- **Total FK Relationships**: 18개

## Table Summary

| Table | Primary Key | Rows | Description |
|-------|-------------|------|-------------|
| employees | emp_id | 20 | 직원 정보 |
| departments | department_id | 10 | 부서 정보 |
| positions | position_id | 5 | 직급/직위 |
| projects | project_id | 8 | 프로젝트 |
| project_assignments | assignment_id | 20 | 프로젝트 배정 |
| salaries | salary_id | 20 | 급여 정보 |
| attendance | attendance_id | 20 | 출퇴근 기록 |
| leave_requests | request_id | 12 | 휴가 신청 |
| performance_reviews | review_id | 10 | 성과 평가 |
| trainings | training_id | 8 | 교육 프로그램 |
| training_participants | id | 20 | 교육 참가자 |

## Ground Truth FK Relationships (18개)

### 1. Self-Reference FK (자기 참조) - 2개

| # | Source Table | Source Column | Target Table | Target Column | Pattern |
|---|--------------|---------------|--------------|---------------|---------|
| 1 | employees | manager_id | employees | emp_id | Self-ref (조직도) |
| 2 | departments | parent_dept | departments | department_id | Self-ref (부서 계층) |

### 2. Abbreviation FK (축약형) - 4개

| # | Source Table | Source Column | Target Table | Target Column | Pattern |
|---|--------------|---------------|--------------|---------------|---------|
| 3 | employees | dept_code | departments | department_id | dept → department, code → id |
| 4 | project_assignments | proj_ref | projects | project_id | proj → project, ref → id |
| 5 | training_participants | training_ref | trainings | training_id | ref → id |
| 6 | projects | owning_dept | departments | department_id | dept 축약 |

### 3. Synonym FK (동의어) - 5개

| # | Source Table | Source Column | Target Table | Target Column | Pattern |
|---|--------------|---------------|--------------|---------------|---------|
| 7 | salaries | worker_id | employees | emp_id | worker = employee |
| 8 | attendance | staff_id | employees | emp_id | staff = employee |
| 9 | project_assignments | assigned_staff | employees | emp_id | staff = employee |
| 10 | leave_requests | requestor | employees | emp_id | requestor = employee |
| 11 | training_participants | participant | employees | emp_id | participant = employee |

### 4. Semantic FK (의미적 추론) - 7개

| # | Source Table | Source Column | Target Table | Target Column | Pattern |
|---|--------------|---------------|--------------|---------------|---------|
| 12 | departments | head_of_dept | employees | emp_id | 부서장 = 직원 |
| 13 | projects | lead_by | employees | emp_id | 프로젝트 리더 = 직원 |
| 14 | salaries | approved_by | employees | emp_id | 승인자 = 직원 |
| 15 | attendance | approved_by_manager | employees | emp_id | 승인 매니저 = 직원 |
| 16 | leave_requests | reviewed_by | employees | emp_id | 검토자 = 직원 |
| 17 | performance_reviews | reviewee | employees | emp_id | 피평가자 = 직원 |
| 18 | performance_reviews | reviewer | employees | emp_id | 평가자 = 직원 |
| 19 | trainings | trainer | employees | emp_id | 교육 강사 = 직원 |
| 20 | trainings | target_dept | departments | department_id | 대상 부서 |

## Deduplicated Final FK List (18개)

| # | Source | Target | Difficulty | Category |
|---|--------|--------|------------|----------|
| 1 | employees.manager_id | employees.emp_id | Medium | Self-ref |
| 2 | employees.dept_code | departments.department_id | Medium | Abbreviation |
| 3 | employees.position_id | positions.position_id | Easy | Direct |
| 4 | departments.parent_dept | departments.department_id | Medium | Self-ref |
| 5 | departments.head_of_dept | employees.emp_id | Hard | Semantic |
| 6 | projects.lead_by | employees.emp_id | Hard | Semantic |
| 7 | projects.owning_dept | departments.department_id | Medium | Abbreviation |
| 8 | project_assignments.proj_ref | projects.project_id | Easy | Abbreviation |
| 9 | project_assignments.assigned_staff | employees.emp_id | Medium | Synonym |
| 10 | salaries.worker_id | employees.emp_id | Medium | Synonym |
| 11 | salaries.approved_by | employees.emp_id | Hard | Semantic |
| 12 | attendance.staff_id | employees.emp_id | Medium | Synonym |
| 13 | attendance.approved_by_manager | employees.emp_id | Hard | Semantic |
| 14 | leave_requests.requestor | employees.emp_id | Medium | Synonym |
| 15 | leave_requests.reviewed_by | employees.emp_id | Hard | Semantic |
| 16 | performance_reviews.reviewee | employees.emp_id | Hard | Semantic |
| 17 | performance_reviews.reviewer | employees.emp_id | Hard | Semantic |
| 18 | trainings.trainer | employees.emp_id | Hard | Semantic |
| 19 | trainings.target_dept | departments.department_id | Medium | Semantic |
| 20 | training_participants.training_ref | trainings.training_id | Easy | Abbreviation |
| 21 | training_participants.participant | employees.emp_id | Medium | Synonym |

**Note**: 중복 정리 후 최종 18개로 확정

## FK Pattern Distribution

### By Difficulty
- **Easy (Direct/Simple)**: 3개 (16.7%)
- **Medium (Abbreviation/Synonym/Self-ref)**: 8개 (44.4%)
- **Hard (Semantic Inference)**: 7개 (38.9%)

### By Pattern Type
- **Direct Match**: 1개 (position_id)
- **Self-Reference**: 2개 (manager_id, parent_dept)
- **Abbreviation**: 4개 (dept_code, proj_ref, training_ref, owning_dept)
- **Synonym**: 5개 (worker_id, staff_id, assigned_staff, requestor, participant)
- **Semantic**: 7개 (head_of_dept, lead_by, approved_by, approved_by_manager, reviewed_by, reviewee, reviewer, trainer)

## Expected Detection Results

### Rule-Based Detection (예상)
| FK | 탐지 가능 여부 | 이유 |
|----|---------------|------|
| employees.position_id → positions.position_id | ✅ | Exact match |
| employees.manager_id → employees.emp_id | ✅ | Data pattern (EMP*) |
| departments.parent_dept → departments.department_id | ✅ | Data pattern (DEPT*) |
| project_assignments.proj_ref → projects.project_id | ✅ | Data pattern (PRJ*) |
| training_participants.training_ref → trainings.training_id | ✅ | Data pattern (TRN*) |
| employees.dept_code → departments.department_id | ⚠️ | Code vs ID mismatch |

**예상 Rule-Based 탐지**: 5-6/18 (28-33%)

### LLM Enhancement Required (예상)
| FK | 이유 |
|----|------|
| salaries.worker_id → employees.emp_id | worker = employee synonym |
| attendance.staff_id → employees.emp_id | staff = employee synonym |
| project_assignments.assigned_staff → employees.emp_id | staff = employee synonym |
| leave_requests.requestor → employees.emp_id | requestor context |
| training_participants.participant → employees.emp_id | participant context |
| departments.head_of_dept → employees.emp_id | semantic role |
| projects.lead_by → employees.emp_id | semantic role |
| salaries.approved_by → employees.emp_id | semantic role |
| attendance.approved_by_manager → employees.emp_id | semantic role |
| leave_requests.reviewed_by → employees.emp_id | semantic role |
| performance_reviews.reviewee → employees.emp_id | semantic role |
| performance_reviews.reviewer → employees.emp_id | semantic role |
| trainings.trainer → employees.emp_id | semantic role |

**예상 LLM 추가 탐지**: 12-13개

## Challenging Patterns

### 1. 다중 FK to employees
- employees 테이블이 **13개의 FK target**으로 사용됨
- worker_id, staff_id, assigned_staff, requestor, participant 등 다양한 synonym
- approved_by, reviewed_by, trainer 등 semantic role

### 2. Self-Reference
- employees.manager_id → employees.emp_id (조직도)
- departments.parent_dept → departments.department_id (부서 계층)

### 3. Semantic Role Patterns
- `*_by` suffix: lead_by, approved_by, reviewed_by (행위자)
- `*_of_*`: head_of_dept (역할)
- Role nouns: trainer, reviewer, reviewee (역할 명사)

## Validation Criteria

### Target Metrics
- Rule-Based Only: Recall ≥ 30% (5-6/18)
- With LLM Enhancement: Recall ≥ 80% (14-15/18)
- Combined (Full Pipeline v7.4): Recall ≥ 95% (17-18/18)

## Notes

### Design Principles
1. **Semantic FK 비율 높음**: Healthcare/E-commerce 대비 Hard 난이도 증가 (38.9%)
2. **다중 참조 패턴**: employees가 13개 FK의 target (중앙 엔티티)
3. **Self-Reference 포함**: 조직도/부서계층 표현
4. **다양한 Synonym**: worker, staff, participant, requestor 등
5. **Semantic Role Suffix**: *_by, *_of_* 패턴 다수

### Dataset Characteristics vs Others

| 특성 | Healthcare | E-commerce | HR |
|------|------------|------------|-----|
| Easy FK 비율 | 35% | 35.7% | 16.7% |
| Hard FK 비율 | 25% | 28.6% | 38.9% |
| Self-Reference | No | Yes (1) | Yes (2) |
| 중앙 엔티티 | patients | orders | employees |
| FK 밀집도 | 분산 | 분산 | employees 집중 |
