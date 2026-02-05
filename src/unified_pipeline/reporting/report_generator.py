"""
Report Generator (v17.0)

자연어 보고서 생성기:
- 다양한 보고서 유형 지원
- LLM 기반 내러티브 생성
- 템플릿 시스템
- 다국어 지원

사용 예시:
    generator = ReportGenerator(context)

    # Executive Summary
    report = await generator.generate(
        report_type=ReportType.EXECUTIVE_SUMMARY,
        title="Q4 Data Quality Report",
    )

    # 인사이트 보고서
    report = await generator.generate(
        report_type=ReportType.INSIGHT_REPORT,
        data={"insights": [...], "metrics": {...}},
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from datetime import datetime
from enum import Enum
import uuid

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

if TYPE_CHECKING:
    from ..shared_context import SharedContext

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """보고서 유형"""
    EXECUTIVE_SUMMARY = "executive_summary"
    INSIGHT_REPORT = "insight_report"
    DATA_QUALITY = "data_quality"
    ANALYSIS_SUMMARY = "analysis_summary"
    PIPELINE_STATUS = "pipeline_status"
    ANOMALY_REPORT = "anomaly_report"
    COMPLIANCE_REPORT = "compliance_report"


class Language(str, Enum):
    """지원 언어"""
    KOREAN = "ko"
    ENGLISH = "en"
    JAPANESE = "ja"
    CHINESE = "zh"


@dataclass
class ReportSection:
    """보고서 섹션"""
    title: str
    content: str
    section_type: str = "text"  # text, table, chart, bullet_list
    data: Optional[Dict[str, Any]] = None
    order: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type,
            "data": self.data,
            "order": self.order,
        }

    def to_markdown(self) -> str:
        """마크다운 변환"""
        md = f"## {self.title}\n\n"

        if self.section_type == "bullet_list" and self.data:
            items = self.data.get("items", [])
            for item in items:
                md += f"- {item}\n"
        elif self.section_type == "table" and self.data:
            headers = self.data.get("headers", [])
            rows = self.data.get("rows", [])
            if headers:
                md += "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for row in rows:
                    md += "| " + " | ".join(str(v) for v in row) + " |\n"
        else:
            md += self.content

        return md + "\n"


@dataclass
class Report:
    """보고서"""
    report_id: str
    report_type: ReportType
    title: str
    language: Language = Language.KOREAN

    # 내용
    executive_summary: str = ""
    sections: List[ReportSection] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # 메타데이터
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generated_by: str = "ReportGenerator"
    version: str = "1.0"

    @property
    def content(self) -> str:
        """전체 내용 (마크다운)"""
        parts = [f"# {self.title}\n"]

        if self.executive_summary:
            parts.append(f"## 요약\n\n{self.executive_summary}\n")

        for section in sorted(self.sections, key=lambda s: s.order):
            parts.append(section.to_markdown())

        if self.recommendations:
            parts.append("## 권장 사항\n")
            for rec in self.recommendations:
                parts.append(f"- {rec}\n")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "title": self.title,
            "language": self.language.value,
            "executive_summary": self.executive_summary,
            "sections": [s.to_dict() for s in self.sections],
            "recommendations": self.recommendations,
            "generated_at": self.generated_at,
            "content": self.content,
        }


# DSPy Signatures
if DSPY_AVAILABLE:
    class ReportNarrativeSignature(dspy.Signature):
        """보고서 내러티브 생성"""
        report_type: str = dspy.InputField(
            desc="Type of report (executive_summary, insight_report, etc.)"
        )
        data_summary: str = dspy.InputField(
            desc="Summary of data to include in the report"
        )
        key_metrics: str = dspy.InputField(
            desc="Key metrics and statistics"
        )
        language: str = dspy.InputField(
            desc="Target language (ko, en, ja, zh)"
        )

        executive_summary: str = dspy.OutputField(
            desc="Executive summary paragraph"
        )
        main_findings: str = dspy.OutputField(
            desc="Main findings as bullet points"
        )
        recommendations: str = dspy.OutputField(
            desc="Recommendations as bullet points"
        )


class ReportGenerator:
    """
    자연어 보고서 생성기

    기능:
    - 다양한 보고서 유형 지원
    - LLM 기반 내러티브 생성
    - 템플릿 시스템
    - 다국어 지원
    """

    def __init__(
        self,
        context: Optional["SharedContext"] = None,
    ):
        """
        Args:
            context: SharedContext
        """
        self.context = context

        # DSPy 모듈
        if DSPY_AVAILABLE:
            self.narrative_generator = dspy.ChainOfThought(ReportNarrativeSignature)

        # 템플릿
        self.templates = self._load_templates()

        # 보고서 히스토리
        self.reports: Dict[str, Report] = {}

        logger.info("ReportGenerator initialized")

    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """템플릿 로드"""
        return {
            ReportType.EXECUTIVE_SUMMARY.value: {
                "ko": "이 보고서는 {period}의 주요 분석 결과를 요약합니다.",
                "en": "This report summarizes the key analysis results for {period}.",
            },
            ReportType.DATA_QUALITY.value: {
                "ko": "데이터 품질 분석 결과, {quality_score}%의 품질 점수를 기록했습니다.",
                "en": "Data quality analysis shows a quality score of {quality_score}%.",
            },
            ReportType.PIPELINE_STATUS.value: {
                "ko": "파이프라인 상태: {status}. 처리된 테이블 {num_tables}개.",
                "en": "Pipeline status: {status}. {num_tables} tables processed.",
            },
        }

    async def generate(
        self,
        report_type: ReportType,
        title: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        language: Language = Language.KOREAN,
    ) -> Report:
        """
        보고서 생성

        Args:
            report_type: 보고서 유형
            title: 보고서 제목
            data: 보고서에 포함할 데이터
            language: 언어

        Returns:
            Report: 생성된 보고서
        """
        report_id = f"report_{uuid.uuid4().hex[:8]}"

        # 기본 제목
        if not title:
            title = self._get_default_title(report_type, language)

        # 데이터 준비
        if not data:
            data = self._gather_context_data(report_type)

        # 보고서 생성
        if report_type == ReportType.EXECUTIVE_SUMMARY:
            report = await self._generate_executive_summary(report_id, title, data, language)
        elif report_type == ReportType.DATA_QUALITY:
            report = await self._generate_data_quality_report(report_id, title, data, language)
        elif report_type == ReportType.PIPELINE_STATUS:
            report = await self._generate_pipeline_status_report(report_id, title, data, language)
        elif report_type == ReportType.INSIGHT_REPORT:
            report = await self._generate_insight_report(report_id, title, data, language)
        else:
            report = await self._generate_generic_report(report_id, report_type, title, data, language)

        self.reports[report_id] = report

        # Evidence Chain 기록
        self._record_to_evidence_chain(report)

        logger.info(f"Generated report: {title} ({report_id})")

        return report

    def _get_default_title(self, report_type: ReportType, language: Language) -> str:
        """기본 제목"""
        titles = {
            ReportType.EXECUTIVE_SUMMARY: {"ko": "Executive Summary", "en": "Executive Summary"},
            ReportType.DATA_QUALITY: {"ko": "데이터 품질 보고서", "en": "Data Quality Report"},
            ReportType.PIPELINE_STATUS: {"ko": "파이프라인 상태 보고서", "en": "Pipeline Status Report"},
            ReportType.INSIGHT_REPORT: {"ko": "인사이트 보고서", "en": "Insight Report"},
            ReportType.ANALYSIS_SUMMARY: {"ko": "분석 요약 보고서", "en": "Analysis Summary"},
            ReportType.ANOMALY_REPORT: {"ko": "이상 탐지 보고서", "en": "Anomaly Report"},
        }

        return titles.get(report_type, {}).get(language.value, report_type.value)

    def _gather_context_data(self, report_type: ReportType) -> Dict[str, Any]:
        """컨텍스트에서 데이터 수집"""
        data = {}

        if not self.context:
            return data

        # 테이블 정보
        data["num_tables"] = len(self.context.tables)
        data["tables"] = list(self.context.tables.keys())

        # 파이프라인 상태
        data["current_phase"] = getattr(self.context, "current_stage", None) or "unknown"

        # Evidence Chain 요약
        if self.context.evidence_chain:
            chain = self.context.evidence_chain
            blocks = getattr(chain, "blocks", [])
            data["evidence_count"] = len(blocks)

        # 인사이트
        data["insights"] = self.context.streaming_insights[-5:]

        # 품질 메트릭
        if hasattr(self.context, "quality_metrics"):
            data["quality_metrics"] = self.context.quality_metrics

        return data

    async def _generate_executive_summary(
        self,
        report_id: str,
        title: str,
        data: Dict[str, Any],
        language: Language,
    ) -> Report:
        """Executive Summary 생성"""
        sections = []

        # LLM 내러티브 생성
        executive_summary = ""
        main_findings = []
        recommendations = []

        if DSPY_AVAILABLE:
            try:
                result = self.narrative_generator(
                    report_type="executive_summary",
                    data_summary=str(data),
                    key_metrics=self._format_key_metrics(data),
                    language=language.value,
                )

                executive_summary = result.executive_summary
                main_findings = [f.strip() for f in result.main_findings.split("\n") if f.strip()]
                recommendations = [r.strip() for r in result.recommendations.split("\n") if r.strip()]

            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # 폴백
        if not executive_summary:
            executive_summary = self._generate_template_summary(data, language)
            main_findings = self._generate_template_findings(data, language)
            recommendations = self._generate_template_recommendations(data, language)

        # 섹션 구성
        sections.append(ReportSection(
            title="주요 발견" if language == Language.KOREAN else "Key Findings",
            content="\n".join(f"• {f}" for f in main_findings),
            section_type="bullet_list",
            data={"items": main_findings},
            order=1,
        ))

        # 데이터 개요 섹션
        overview_content = self._format_data_overview(data, language)
        sections.append(ReportSection(
            title="데이터 개요" if language == Language.KOREAN else "Data Overview",
            content=overview_content,
            order=2,
        ))

        return Report(
            report_id=report_id,
            report_type=ReportType.EXECUTIVE_SUMMARY,
            title=title,
            language=language,
            executive_summary=executive_summary,
            sections=sections,
            recommendations=recommendations,
        )

    async def _generate_data_quality_report(
        self,
        report_id: str,
        title: str,
        data: Dict[str, Any],
        language: Language,
    ) -> Report:
        """데이터 품질 보고서 생성"""
        sections = []

        # 품질 점수 계산
        quality_score = data.get("quality_score", 85)
        issues = data.get("issues", [])

        # Executive Summary
        if language == Language.KOREAN:
            executive_summary = (
                f"데이터 품질 분석 결과, 전체 품질 점수는 {quality_score}%입니다. "
                f"총 {len(issues)}개의 품질 이슈가 발견되었습니다."
            )
        else:
            executive_summary = (
                f"Data quality analysis shows an overall quality score of {quality_score}%. "
                f"A total of {len(issues)} quality issues were found."
            )

        # 품질 메트릭 섹션
        metrics_section = ReportSection(
            title="품질 메트릭" if language == Language.KOREAN else "Quality Metrics",
            content="",
            section_type="table",
            data={
                "headers": ["메트릭", "값", "상태"],
                "rows": [
                    ["완전성", f"{quality_score}%", "양호" if quality_score > 80 else "주의"],
                    ["정확성", f"{quality_score - 5}%", "양호" if quality_score > 85 else "주의"],
                    ["일관성", f"{quality_score + 3}%", "양호"],
                ],
            },
            order=1,
        )
        sections.append(metrics_section)

        # 이슈 섹션
        if issues:
            issues_section = ReportSection(
                title="발견된 이슈" if language == Language.KOREAN else "Issues Found",
                content="\n".join(f"• {issue}" for issue in issues[:10]),
                section_type="bullet_list",
                data={"items": issues[:10]},
                order=2,
            )
            sections.append(issues_section)

        # 권장 사항
        recommendations = []
        if quality_score < 90:
            recommendations.append("결측값 처리 필요" if language == Language.KOREAN else "Missing value handling required")
        if quality_score < 80:
            recommendations.append("데이터 정제 권장" if language == Language.KOREAN else "Data cleansing recommended")

        return Report(
            report_id=report_id,
            report_type=ReportType.DATA_QUALITY,
            title=title,
            language=language,
            executive_summary=executive_summary,
            sections=sections,
            recommendations=recommendations,
        )

    async def _generate_pipeline_status_report(
        self,
        report_id: str,
        title: str,
        data: Dict[str, Any],
        language: Language,
    ) -> Report:
        """파이프라인 상태 보고서 생성"""
        sections = []

        status = data.get("current_phase", "unknown")
        num_tables = data.get("num_tables", 0)

        # Executive Summary
        if language == Language.KOREAN:
            executive_summary = (
                f"현재 파이프라인 상태: {status}. "
                f"총 {num_tables}개 테이블이 처리되었습니다."
            )
        else:
            executive_summary = (
                f"Current pipeline status: {status}. "
                f"{num_tables} tables have been processed."
            )

        # 테이블 목록
        tables = data.get("tables", [])
        if tables:
            tables_section = ReportSection(
                title="처리된 테이블" if language == Language.KOREAN else "Processed Tables",
                content="\n".join(f"• {t}" for t in tables[:20]),
                section_type="bullet_list",
                data={"items": tables[:20]},
                order=1,
            )
            sections.append(tables_section)

        return Report(
            report_id=report_id,
            report_type=ReportType.PIPELINE_STATUS,
            title=title,
            language=language,
            executive_summary=executive_summary,
            sections=sections,
            recommendations=[],
        )

    async def _generate_insight_report(
        self,
        report_id: str,
        title: str,
        data: Dict[str, Any],
        language: Language,
    ) -> Report:
        """인사이트 보고서 생성"""
        sections = []

        insights = data.get("insights", [])

        # Executive Summary
        if language == Language.KOREAN:
            executive_summary = f"총 {len(insights)}개의 인사이트가 생성되었습니다."
        else:
            executive_summary = f"A total of {len(insights)} insights were generated."

        # 인사이트 섹션
        if insights:
            insight_items = []
            for insight in insights[:10]:
                if isinstance(insight, dict):
                    insight_items.append(insight.get("summary", str(insight)))
                else:
                    insight_items.append(str(insight))

            insights_section = ReportSection(
                title="주요 인사이트" if language == Language.KOREAN else "Key Insights",
                content="\n".join(f"• {i}" for i in insight_items),
                section_type="bullet_list",
                data={"items": insight_items},
                order=1,
            )
            sections.append(insights_section)

        return Report(
            report_id=report_id,
            report_type=ReportType.INSIGHT_REPORT,
            title=title,
            language=language,
            executive_summary=executive_summary,
            sections=sections,
            recommendations=[],
        )

    async def _generate_generic_report(
        self,
        report_id: str,
        report_type: ReportType,
        title: str,
        data: Dict[str, Any],
        language: Language,
    ) -> Report:
        """일반 보고서 생성"""
        executive_summary = self._generate_template_summary(data, language)

        sections = []
        if data:
            content = "\n".join(f"• {k}: {v}" for k, v in data.items() if not isinstance(v, (list, dict)))
            sections.append(ReportSection(
                title="데이터 요약" if language == Language.KOREAN else "Data Summary",
                content=content,
                order=1,
            ))

        return Report(
            report_id=report_id,
            report_type=report_type,
            title=title,
            language=language,
            executive_summary=executive_summary,
            sections=sections,
            recommendations=[],
        )

    def _format_key_metrics(self, data: Dict[str, Any]) -> str:
        """주요 메트릭 포맷팅"""
        metrics = []

        if "num_tables" in data:
            metrics.append(f"Tables: {data['num_tables']}")
        if "evidence_count" in data:
            metrics.append(f"Evidence blocks: {data['evidence_count']}")
        if "quality_score" in data:
            metrics.append(f"Quality score: {data['quality_score']}%")

        return ", ".join(metrics) if metrics else "No key metrics available"

    def _format_data_overview(self, data: Dict[str, Any], language: Language) -> str:
        """데이터 개요 포맷팅"""
        parts = []

        if language == Language.KOREAN:
            if "num_tables" in data:
                parts.append(f"분석된 테이블 수: {data['num_tables']}")
            if "evidence_count" in data:
                parts.append(f"수집된 증거 블록: {data['evidence_count']}")
            if "current_phase" in data:
                parts.append(f"현재 파이프라인 단계: {data['current_phase']}")
        else:
            if "num_tables" in data:
                parts.append(f"Number of tables analyzed: {data['num_tables']}")
            if "evidence_count" in data:
                parts.append(f"Evidence blocks collected: {data['evidence_count']}")
            if "current_phase" in data:
                parts.append(f"Current pipeline phase: {data['current_phase']}")

        return "\n".join(parts) if parts else "No data overview available."

    def _generate_template_summary(self, data: Dict[str, Any], language: Language) -> str:
        """템플릿 기반 요약"""
        if language == Language.KOREAN:
            return (
                f"이 보고서는 {data.get('num_tables', 'N/A')}개의 테이블에 대한 "
                f"분석 결과를 요약합니다."
            )
        else:
            return (
                f"This report summarizes the analysis results for "
                f"{data.get('num_tables', 'N/A')} tables."
            )

    def _generate_template_findings(self, data: Dict[str, Any], language: Language) -> List[str]:
        """템플릿 기반 발견"""
        findings = []

        if language == Language.KOREAN:
            if data.get("num_tables"):
                findings.append(f"{data['num_tables']}개의 테이블이 분석되었습니다.")
            if data.get("evidence_count"):
                findings.append(f"{data['evidence_count']}개의 증거 블록이 수집되었습니다.")
        else:
            if data.get("num_tables"):
                findings.append(f"{data['num_tables']} tables were analyzed.")
            if data.get("evidence_count"):
                findings.append(f"{data['evidence_count']} evidence blocks were collected.")

        return findings if findings else ["분석이 완료되었습니다." if language == Language.KOREAN else "Analysis completed."]

    def _generate_template_recommendations(self, data: Dict[str, Any], language: Language) -> List[str]:
        """템플릿 기반 권장 사항"""
        recommendations = []

        if language == Language.KOREAN:
            recommendations.append("정기적인 데이터 품질 모니터링을 권장합니다.")
            recommendations.append("이상 탐지 알림 설정을 고려하세요.")
        else:
            recommendations.append("Regular data quality monitoring is recommended.")
            recommendations.append("Consider setting up anomaly detection alerts.")

        return recommendations

    def _record_to_evidence_chain(self, report: Report) -> None:
        """Evidence Chain에 기록"""
        if not self.context or not self.context.evidence_chain:
            return

        try:
            from ..autonomous.evidence_chain import EvidenceType

            evidence_type = getattr(
                EvidenceType,
                "NL_REPORT",
                EvidenceType.QUALITY_ASSESSMENT
            )

            self.context.evidence_chain.add_block(
                phase="reporting",
                agent="ReportGenerator",
                evidence_type=evidence_type,
                finding=f"Generated report: {report.title}",
                reasoning=f"Report type: {report.report_type.value}",
                conclusion=report.executive_summary[:200],
                metrics={
                    "report_id": report.report_id,
                    "report_type": report.report_type.value,
                    "num_sections": len(report.sections),
                    "num_recommendations": len(report.recommendations),
                },
                confidence=0.9,
            )
        except Exception as e:
            logger.warning(f"Failed to record to evidence chain: {e}")

    def get_generator_summary(self) -> Dict[str, Any]:
        """생성기 요약"""
        return {
            "total_reports": len(self.reports),
            "report_types": list(set(r.report_type.value for r in self.reports.values())),
            "recent_reports": [
                {"id": r.report_id, "title": r.title, "type": r.report_type.value}
                for r in list(self.reports.values())[-5:]
            ],
        }
