# Airport Data Silo - Analysis Answer Key

이 문서는 `airport_silo` 데이터셋의 정답지입니다. Ontoloty 시스템이 이 데이터를 분석했을 때 도출해야 하는 올바른 결과를 기술합니다.

---

## 1. 데이터셋 구조

### 1.1 파일 목록 (8개 CSV)
| 파일명 | 레코드 수 | 설명 |
|--------|----------|------|
| flights.csv | 100 | 항공편 정보 |
| passengers.csv | 120 | 승객 정보 |
| airlines.csv | 30 | 항공사 마스터 데이터 |
| gates.csv | 52 | 게이트 정보 |
| delay_records.csv | 51 | 지연 기록 |
| baggage_handling.csv | 157 | 수하물 처리 |
| airport_employees.csv | 70 | 공항 직원 정보 |
| aircraft_maintenance.csv | 40 | 항공기 정비 기록 |
| aircraft_registry.csv | 48 | 항공기 등록 정보 |

---

## 2. 도메인 감지 (Domain Detection)

### 2.1 Expected Domain
- **Primary Domain**: Aviation / Airport Operations
- **Confidence**: 0.95+

### 2.2 Domain Indicators
- 항공 관련 용어: flight_number, carrier_code, aircraft_type, gate, IATA_code, ICAO_code
- 공항 관련 용어: terminal, concourse, baggage, boarding, check_in
- 항공사 코드: KE, OZ, 7C, LJ, TW, ZE (Korean carriers)
- 공항 코드: ICN (Incheon), LAX, NRT, SIN 등

---

## 3. 엔티티 분류 (Entity Classification)

### 3.1 Core Entities

| Entity | Source Table(s) | Primary Key | Classification |
|--------|-----------------|-------------|----------------|
| Flight | flights.csv | flight_id | **TRANSACTIONAL** |
| Passenger | passengers.csv | pax_id | **TRANSACTIONAL** |
| Airline | airlines.csv | airline_id | **MASTER** |
| Gate | gates.csv | gate_id | **MASTER** |
| Employee | airport_employees.csv | emp_id | **MASTER** |
| Aircraft | aircraft_registry.csv | registration_no | **MASTER** |
| Delay | delay_records.csv | record_id | **EVENT** |
| Baggage | baggage_handling.csv | bag_tag_id | **TRANSACTIONAL** |
| Maintenance | aircraft_maintenance.csv | maintenance_id | **EVENT** |

### 3.2 Entity Hierarchy
```
Airport (Implied Root)
├── Terminal
│   ├── Concourse
│   │   └── Gate
│   └── Belt (Baggage)
├── Airline
│   ├── Aircraft
│   └── Flight
│       ├── Passenger
│       │   └── Baggage
│       └── Delay
├── Employee
│   └── Department
└── Maintenance
    └── Work Order
```

---

## 4. 관계 발견 (Relationship Discovery)

### 4.1 명시적 관계 (Explicit - Foreign Key)

| From Table | To Table | Join Keys | Relationship Type |
|------------|----------|-----------|-------------------|
| flights | airlines | carrier_code = iata_code | N:1 |
| flights | gates | gate_number = gate_code | N:1 |
| passengers | flights | flight_num = flight_number | N:1 |
| baggage_handling | passengers | booking_reference = booking_ref | N:1 |
| baggage_handling | flights | flt_number = flight_number | N:1 |
| aircraft_maintenance | aircraft_registry | aircraft_reg = registration_no | N:1 |
| aircraft_maintenance | airport_employees | mechanic_id = emp_id | N:1 |
| delay_records | flights | flt_no = flight_number | N:1 |

### 4.2 암시적 관계 (Implicit - 데이터 사일로 문제)

⚠️ **Critical Discovery**: 아래 관계들은 명시적 FK가 없지만 데이터 분석을 통해 발견해야 함

| Relationship | Evidence | Join Logic |
|--------------|----------|------------|
| airlines ↔ delay_records | `carrier` 컬럼이 `icao_code` 사용 | carrier = icao_code (NOT iata_code!) |
| airlines ↔ aircraft_registry | `operator_code` = `iata_code` | Different column names |
| airlines ↔ aircraft_maintenance | `airline` = `iata_code` | Abbreviated reference |
| gates ↔ airlines | `assigned_airline` = `iata_code` | Gate assignment |
| employees ↔ maintenance | `supervisor_id` references `emp_id` | Self-reference |
| flights ↔ aircraft_registry | **NO DIRECT LINK** | Must infer from aircraft_type + airline |

### 4.3 데이터 사일로 불일치 (Key Findings)

1. **항공사 코드 불일치**
   - `flights.csv`: `carrier_code` (IATA: KE, OZ)
   - `delay_records.csv`: `carrier` (ICAO: KAL, AAR)
   - `airlines.csv`: 둘 다 존재 (`iata_code`, `icao_code`)

2. **네이밍 불일치**
   - Flight identifier: `flight_id`, `flight_number`, `flt_no`, `flt_number`, `flight_num`
   - Passenger: `pax_id`, `passenger_name`, `booking_reference`, `booking_ref`

3. **날짜 형식 일관성** (Good - 모두 ISO 8601)

---

## 5. 데이터 품질 이슈

### 5.1 Missing Values
| Table | Column | Missing Count | Impact |
|-------|--------|---------------|--------|
| flights | actual_departure | 10 | SCHEDULED/DELAYED flights |
| flights | actual_arrival | 10 | SCHEDULED/DELAYED flights |
| delay_records | resolution_time | 2 | Ongoing delays |
| baggage_handling | check_in_timestamp | 2 | NOT_CHECKED_IN passengers |
| baggage_handling | weight_kg | 2 | Not yet checked |

### 5.2 Data Quality Issues to Detect

1. **Cancelled Flight**: TW999 (FL087) - `CANCELLED` status, baggage stranded
2. **Delayed Flight**: LJ999 (FL085) - weather delay (45 min)
3. **Grounded Aircraft**: HL7705 - hydraulic failure, TW999 사용 불가
4. **Damaged Baggage**: BT2401150023 (Maria Garcia) - `damage_reported=TRUE, claim_filed=TRUE`
5. **Baggage Anomaly**: BT2401160014 (Emma Wilson) - damaged but no claim filed
6. **Gate Maintenance**: A14, B16, C16 - status=MAINTENANCE

### 5.3 Orphan Records
- No orphan records by design (all FKs valid)

---

## 6. 통계적 인사이트 (Expected Insights)

### 6.1 Flight Statistics
- **Total Flights**: 100
- **Arrived**: 80
- **Scheduled**: 16
- **Delayed**: 1
- **Cancelled**: 1
- **Boarding**: 2

### 6.2 Passenger Statistics
- **Total Passengers**: 120
- **Boarded**: 102
- **Checked In**: 12
- **Not Checked In**: 4
- **Cancelled**: 2

### 6.3 Delay Analysis
- **Total Delays**: 51 records
- **By Category**:
  - MINIMAL (0-5min): 21
  - MINOR (6-15min): 24
  - MODERATE (16-30min): 5
  - SIGNIFICANT (31+min): 1
  - CANCELLED: 1

- **By Responsible Party**:
  - AIRLINE: 16 (31%)
  - PASSENGER: 16 (31%)
  - AIRPORT: 6 (12%)
  - GROUND_HANDLER: 4 (8%)
  - ATC: 4 (8%)
  - SECURITY: 4 (8%)
  - WEATHER: 1 (2%)

- **By Delay Code**:
  - 11 (LATE_PAX): 16
  - 81 (LATE_CREW): 13
  - 63 (CONNECTING_PAX): 8
  - 72 (WEIGHT_BALANCE): 5
  - 41 (FUELING): 4
  - 71 (ATC_DELAY): 4
  - 93 (SECURITY): 4
  - 74 (AIRCRAFT_CHANGE): 1
  - 77 (WEATHER): 1
  - 99 (AIRCRAFT_ISSUE): 1

### 6.4 Airline Performance
| Airline | Total Flights | Delays | Delay Rate |
|---------|--------------|--------|------------|
| KE (Korean Air) | 40 | 22 | 55% |
| OZ (Asiana) | 24 | 12 | 50% |
| 7C (Jeju Air) | 14 | 8 | 57% |
| LJ (Jin Air) | 10 | 7 | 70% |
| TW (T'way) | 8 | 5 | 63% |
| ZE (Eastar) | 8 | 6 | 75% |

### 6.5 Baggage Statistics
- **Total Bags**: 157
- **Delivered**: 143
- **Pending**: 8
- **Delayed**: 3
- **In Transit**: 3
- **Damaged**: 2
- **Claims Filed**: 1

### 6.6 Maintenance Statistics
- **Total Records**: 40
- **Completed**: 34
- **In Progress**: 5
- **Pending Parts**: 1

- **By Type**:
  - TRANSIT_CHECK: 12
  - LINE_CHECK: 15
  - A_CHECK: 4
  - B_CHECK: 2
  - C_CHECK: 1
  - ENGINE_INSPECTION: 1
  - AVIONICS_UPDATE: 1
  - UNSCHEDULED: 1
  - GROUNDED: 1

### 6.7 Employee Statistics
- **Total Employees**: 70
- **By Department**:
  - OPERATIONS: 14
  - SECURITY: 8
  - MAINTENANCE: 10
  - CUSTOMER_SERVICE: 7
  - GROUND_HANDLING: 10
  - ATC: 5
  - ADMINISTRATION: 5
  - IT: 5
  - EMERGENCY: 6

- **By Status**:
  - ACTIVE: 68
  - TRAINING: 2

- **By Contract Type**:
  - PERMANENT: 48
  - CONTRACT: 22

---

## 7. 온톨로지 구축 정답

### 7.1 Classes (Expected)
```
Thing
├── Organization
│   ├── Airline
│   └── Airport
├── Infrastructure
│   ├── Terminal
│   ├── Gate
│   ├── Hangar
│   └── Belt
├── Vehicle
│   └── Aircraft
├── Person
│   ├── Passenger
│   └── Employee
│       ├── OperationsStaff
│       ├── SecurityStaff
│       ├── MaintenanceTechnician
│       ├── CustomerServiceRep
│       ├── GroundHandler
│       ├── AirTrafficController
│       └── EmergencyResponder
├── Event
│   ├── Flight
│   ├── Delay
│   ├── MaintenanceEvent
│   └── BaggageEvent
└── Object
    └── Baggage
```

### 7.2 Object Properties (Expected)
```
operatedBy: Flight → Airline
departedFrom: Flight → Gate
arrivedAt: Flight → Gate
usesAircraft: Flight → Aircraft (inferred)
ownedBy: Aircraft → Airline
assignedTo: Gate → Airline
worksIn: Employee → Department
supervisedBy: Employee → Employee
performedMaintenance: MaintenanceTechnician → MaintenanceEvent
bookedOn: Passenger → Flight
belongsTo: Baggage → Passenger
handledAt: Baggage → Flight
delayedFlight: Delay → Flight
```

### 7.3 Data Properties (Expected)
- flightNumber: Flight → xsd:string
- scheduledDeparture: Flight → xsd:dateTime
- delayMinutes: Delay → xsd:integer
- passengerCount: Flight → xsd:integer
- baggageWeight: Baggage → xsd:decimal
- seatClass: Passenger → {F, C, Y}
- maintenanceType: MaintenanceEvent → {TRANSIT_CHECK, LINE_CHECK, A_CHECK, B_CHECK, C_CHECK, ...}

---

## 8. Cross-System Integration Rules

### 8.1 Entity Resolution Rules
| Scenario | Rule |
|----------|------|
| Airline Matching | IATA code OR ICAO code → Same Airline |
| Flight Matching | flight_number + scheduled_date → Same Flight |
| Passenger Matching | booking_ref = pnr_code = booking_reference |
| Aircraft Matching | registration_no (unique identifier) |

### 8.2 Data Transformation Requirements
1. **Normalize Airline Codes**: ICAO → IATA mapping
2. **Standardize Flight IDs**: `flt_no` → `flight_number` format
3. **Unify Timestamps**: All to UTC ISO 8601
4. **Baggage-Passenger Link**: Via booking_reference

---

## 9. 검증 체크리스트

Ontoloty 시스템이 이 데이터를 분석할 때 다음을 검증:

### 9.1 Domain Detection
- [ ] Aviation/Airport 도메인 감지
- [ ] Confidence ≥ 0.90

### 9.2 Entity Discovery
- [ ] 최소 9개 주요 엔티티 식별
- [ ] Master/Transactional/Event 분류 정확

### 9.3 Relationship Discovery
- [ ] 명시적 관계 8개 이상 발견
- [ ] 암시적 관계(IATA↔ICAO) 발견
- [ ] 데이터 사일로 불일치 감지

### 9.4 Data Quality
- [ ] Missing values 감지
- [ ] Cancelled flight(TW999) 식별
- [ ] Grounded aircraft(HL7705) 식별
- [ ] Damaged baggage case 식별

### 9.5 Insights
- [ ] 항공사별 지연율 분석
- [ ] 지연 원인 분포 분석
- [ ] 부서별 직원 분포 분석

---

## 10. 난이도 평가

| 항목 | 난이도 | 설명 |
|------|--------|------|
| Domain Detection | ★☆☆ | 명확한 항공 도메인 |
| Entity Classification | ★★☆ | 다양한 엔티티 타입 |
| Explicit Relationships | ★★☆ | FK 명시적이나 네이밍 불일치 |
| Implicit Relationships | ★★★ | IATA/ICAO 매핑 필요 |
| Data Quality Issues | ★★☆ | 의도적 누락 및 이상 데이터 |
| Cross-System Integration | ★★★ | 데이터 사일로 해결 필요 |

**Overall Difficulty**: ★★☆ (Medium)

---

## 11. 추가 참고 사항

### 11.1 데이터 생성 로직
- 2024-01-15 ~ 2024-01-20 기간 데이터
- 인천국제공항(ICN) 중심
- 한국 국적 항공사 6개 (KE, OZ, 7C, LJ, TW, ZE)
- 실제 항공 업계 표준 코드 및 용어 사용

### 11.2 의도적 데이터 사일로 패턴
1. 동일 엔티티의 다른 식별자 사용 (IATA vs ICAO)
2. 동일 속성의 다른 컬럼명 사용
3. 분리된 시스템에서 관리되는 데이터 시뮬레이션
   - OPS 시스템: flights, passengers, gates
   - BAGGAGE 시스템: baggage_handling
   - DELAY 시스템: delay_records
   - MRO 시스템: aircraft_maintenance, aircraft_registry
   - HR 시스템: airport_employees

### 11.3 정답 활용
이 정답지는 Ontoloty 시스템의:
- 자동화된 테스트 검증
- 분석 정확도 측정
- 데이터 사일로 통합 능력 평가
에 활용될 수 있습니다.

---

*Generated by: Claude (Data Set Creator)*
*Date: 2024-01-20*
*Version: 1.0*
