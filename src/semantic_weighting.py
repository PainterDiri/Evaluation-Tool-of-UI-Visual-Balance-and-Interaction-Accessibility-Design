from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .ui_types import UIElement

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class SemanticWeighter:
    """Assign semantic priority alpha in [0.1,1.0] to interactive elements."""

    def __init__(self, use_llm: bool, model: str, api_key: str, base_url: str):
        self.use_llm = use_llm and OpenAI is not None and bool(api_key)
        self.model = model
        self.client = None
        if self.use_llm:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    def apply(self, elements: list[UIElement], image_source: str | Path | np.ndarray) -> dict[str, Any]:
        interactive = [e for e in elements if e.is_interactive]
        if not interactive:
            return {"mode": "none", "reason": "no interactive elements"}

        if self.use_llm and self.client is not None:
            try:
                payload = self._build_payload(interactive)
                result = self._query_llm(payload, image_source)
                self._merge_weights(interactive, result)
                return {"mode": "llm", "result": result}
            except Exception as exc:
                self._heuristic_assign(interactive)
                return {"mode": "heuristic-fallback", "reason": str(exc)}

        self._heuristic_assign(interactive)
        return {"mode": "heuristic", "reason": "llm disabled or unavailable"}

    def generate_detailed_diagnosis(
        self,
        *,
        image_source: str | Path | np.ndarray,
        width: int,
        height: int,
        physical_balance: float,
        accessibility: float,
        weighted_cost: float,
        center_of_mass: tuple[float, float],
        geometric_center: tuple[float, float],
        origin: tuple[float, float],
        interactive_elements: list[UIElement],
        base_recommendations: list[str],
    ) -> dict[str, Any]:
        if not self.use_llm or self.client is None:
            return {"mode": "skipped", "reason": "llm disabled or unavailable"}

        try:
            payload = self._build_diagnosis_payload(interactive_elements, origin)

            dx = center_of_mass[0] - geometric_center[0]
            dy = center_of_mass[1] - geometric_center[1]
            offset = math.hypot(dx, dy)
            max_dist = math.hypot(geometric_center[0], geometric_center[1])
            offset_ratio = 0.0 if max_dist <= 1e-9 else offset / max_dist

            prompt = (
                "你是移动端UI设计评审专家。请基于输入截图与量化指标，输出系统化、规则化、可执行的中文诊断。"
                "\n必须严格使用JSON对象，不要输出Markdown。"
                "\n诊断时必须明确回答两件事："
                "\n1) 视觉平衡性为什么高/低（基于视觉重心偏移、上下/左右视觉重量分布、关键高对比区域位置）"
                "\n2) 交互可达性为什么高/低（基于Fitts成本、高语义权重元素位置和尺寸、拇指起点距离）"
                "\n再给出非常详细、分优先级、可落地的优化方案。"
                "\n返回JSON结构："
                "\n{"
                "\n  \"overall_summary\": \"...\"," 
                "\n  \"visual_balance\": {"
                "\n    \"level\": \"high|medium|low\"," 
                "\n    \"why\": \"...\"," 
                "\n    \"evidence\": [\"...\", \"...\"],"
                "\n    \"rule_explanation\": \"...\""
                "\n  },"
                "\n  \"interaction_accessibility\": {"
                "\n    \"level\": \"high|medium|low\"," 
                "\n    \"why\": \"...\"," 
                "\n    \"evidence\": [\"...\", \"...\"],"
                "\n    \"rule_explanation\": \"...\""
                "\n  },"
                "\n  \"optimization_plan\": ["
                "\n    {"
                "\n      \"priority\": \"P0|P1|P2\"," 
                "\n      \"title\": \"...\"," 
                "\n      \"target_elements\": [1,2],"
                "\n      \"problem\": \"...\"," 
                "\n      \"design_rule\": \"...\"," 
                "\n      \"actions\": [\"步骤1...\", \"步骤2...\", \"步骤3...\"],"
                "\n      \"expected_effect\": \"...\"," 
                "\n      \"verification\": \"如何验证优化有效\""
                "\n    }"
                "\n  ]"
                "\n}"
                "\n要求：evidence和actions都要具体，不要空泛。若涉及组件，请尽量引用id。"
                f"\n屏幕尺寸: width={width}, height={height}"
                f"\n分数: physical_balance={physical_balance:.4f}, accessibility={accessibility:.4f}, weighted_cost={weighted_cost:.4f}"
                f"\n视觉重心: center_of_mass=({center_of_mass[0]:.2f},{center_of_mass[1]:.2f}), geometric_center=({geometric_center[0]:.2f},{geometric_center[1]:.2f}), offset_ratio={offset_ratio:.4f}"
                f"\n拇指起点: origin=({origin[0]:.2f},{origin[1]:.2f})"
                f"\n交互组件摘要: {json.dumps(payload, ensure_ascii=False)}"
                f"\n现有规则建议(可参考): {json.dumps(base_recommendations, ensure_ascii=False)}"
            )

            image_b64 = self._encode_image(image_source)
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.15,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
            )

            content = response.choices[0].message.content
            raw = json.loads(content)
            return {
                "mode": "llm-detailed",
                "diagnosis": self._normalize_diagnosis(raw),
            }
        except Exception as exc:
            return {"mode": "fallback", "reason": str(exc)}

    def summarize_brand_pattern(
        self,
        *,
        brand_name: str,
        sample_cases: list[dict[str, Any]],
        sample_images: list[str | Path | np.ndarray],
    ) -> dict[str, Any]:
        if not sample_cases:
            return {"mode": "none", "reason": "no sample cases", "summary": {}}

        if not self.use_llm or self.client is None:
            return {
                "mode": "heuristic-summary",
                "reason": "llm disabled or unavailable",
                "summary": self._build_statistical_pattern_summary(brand_name, sample_cases),
            }

        try:
            payload = sample_cases[:8]
            prompt = (
                "你是资深移动端品牌设计研究员。"
                "请结合多张同品牌截图和每张图的量化分数，总结该品牌可复用的UI设计范式。"
                "\n必须严格输出JSON对象，不要输出Markdown。"
                "\n输出结构："
                "\n{"
                "\n  \"brand_positioning\": \"一句话描述该品牌在界面风格与交互上的定位\","
                "\n  \"pattern_summary\": \"整体范式总结（2-4句）\","
                "\n  \"design_patterns\": ["
                "\n    {"
                "\n      \"name\": \"范式名称\","
                "\n      \"rule\": \"可执行设计规则\","
                "\n      \"evidence\": [\"来自哪几张样本的证据\"],"
                "\n      \"quant_signal\": \"与分数/指标的关系\""
                "\n    }"
                "\n  ],"
                "\n  \"score_baseline\": {"
                "\n    \"balance_mean\": 0-10浮点数,"
                "\n    \"accessibility_mean\": 0-10浮点数,"
                "\n    \"final_mean\": 0-10浮点数,"
                "\n    \"consistency\": \"high|medium|low\""
                "\n  },"
                "\n  \"dos\": [\"应坚持的做法\"],"
                "\n  \"donts\": [\"应避免的做法\"],"
                "\n  \"checklist\": [\"评审时可直接核对的检查项\"]"
                "\n}"
                "\n要求："
                "\n1) design_patterns 至少3条，最多6条；"
                "\n2) 每条 pattern 必须包含可执行 rule 与 evidence；"
                "\n3) checklist 优先给可落地、可验证的条目。"
                f"\n品牌名: {brand_name.strip() or '未命名品牌'}"
                f"\n样本统计JSON: {json.dumps(payload, ensure_ascii=False)}"
            )

            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            for idx, image in enumerate(sample_images[:6], start=1):
                image_b64 = self._encode_image(image)
                content.append({"type": "text", "text": f"样本截图#{idx}"})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    }
                )

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": content}],
            )
            content_text = response.choices[0].message.content
            raw = json.loads(content_text)
            return {
                "mode": "llm-brand-pattern",
                "summary": self._normalize_brand_pattern_summary(raw),
            }
        except Exception as exc:
            return {
                "mode": "fallback",
                "reason": str(exc),
                "summary": self._build_statistical_pattern_summary(brand_name, sample_cases),
            }

    def diagnose_against_brand_pattern(
        self,
        *,
        brand_name: str,
        brand_pattern_summary: dict[str, Any],
        candidate_case: dict[str, Any],
        candidate_image: str | Path | np.ndarray,
    ) -> dict[str, Any]:
        heuristic = self._build_heuristic_gap_diagnosis(
            brand_name=brand_name,
            brand_pattern_summary=brand_pattern_summary,
            candidate_case=candidate_case,
        )

        if not self.use_llm or self.client is None:
            return {
                "mode": "heuristic-diagnosis",
                "reason": "llm disabled or unavailable",
                "diagnosis": heuristic,
            }

        try:
            prompt = (
                "你是移动端设计评审专家。"
                "请先根据候选页面评分判断其与品牌范式的匹配度，再基于截图内容指出不足并给出改进建议。"
                "\n必须严格输出JSON对象，不要输出Markdown。"
                "\n输出结构："
                "\n{"
                "\n  \"overall_alignment\": \"high|medium|low\","
                "\n  \"summary\": \"总体判断（2-3句）\","
                "\n  \"score_alignment\": {"
                "\n    \"balance_gap\": 候选-基线,"
                "\n    \"accessibility_gap\": 候选-基线,"
                "\n    \"final_gap\": 候选-基线,"
                "\n    \"comment\": \"分数差距解读\""
                "\n  },"
                "\n  \"violations\": ["
                "\n    {"
                "\n      \"pattern\": \"违反的范式\","
                "\n      \"issue\": \"具体问题\","
                "\n      \"severity\": \"high|medium|low\","
                "\n      \"evidence\": \"来自截图与指标的证据\""
                "\n    }"
                "\n  ],"
                "\n  \"improvements\": ["
                "\n    {"
                "\n      \"priority\": \"P0|P1|P2\","
                "\n      \"title\": \"优化标题\","
                "\n      \"action\": \"可执行动作\","
                "\n      \"expected_gain\": \"预期收益\","
                "\n      \"related_pattern\": \"对应品牌范式\""
                "\n    }"
                "\n  ],"
                "\n  \"quick_wins\": [\"低成本高收益优化\"]"
                "\n}"
                "\n要求："
                "\n1) violations 至少2条，improvements 至少3条；"
                "\n2) improvements 按 priority 从高到低排序；"
                "\n3) 优先关注视觉平衡和交互可达性两个维度。"
                f"\n品牌名: {brand_name.strip() or '未命名品牌'}"
                f"\n品牌范式JSON: {json.dumps(brand_pattern_summary, ensure_ascii=False)}"
                f"\n候选页面JSON: {json.dumps(candidate_case, ensure_ascii=False)}"
            )

            image_b64 = self._encode_image(candidate_image)
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.18,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
            )
            content_text = response.choices[0].message.content
            raw = json.loads(content_text)
            return {
                "mode": "llm-pattern-diagnosis",
                "diagnosis": self._normalize_brand_gap_diagnosis(raw),
            }
        except Exception as exc:
            return {
                "mode": "fallback",
                "reason": str(exc),
                "diagnosis": heuristic,
            }

    def _heuristic_assign(self, elements: list[UIElement]) -> None:
        for e in elements:
            if e.role_guess == "primary_cta":
                e.semantic_weight = 1.0
            elif e.role_guess == "button_or_tab":
                e.semantic_weight = 0.75
            elif e.role_guess == "card":
                e.semantic_weight = 0.55
            elif e.role_guess == "top_nav_or_status":
                e.semantic_weight = 0.35
            else:
                e.semantic_weight = 0.25

    def _build_payload(self, elements: list[UIElement]) -> list[dict[str, Any]]:
        return [
            {
                "id": e.id,
                "bbox": {"x": e.x, "y": e.y, "w": e.w, "h": e.h},
                "role_guess": e.role_guess,
            }
            for e in elements
        ]

    def _build_diagnosis_payload(
        self,
        interactive_elements: list[UIElement],
        origin: tuple[float, float],
        max_items: int = 12,
    ) -> list[dict[str, Any]]:
        scored = sorted(
            interactive_elements,
            key=lambda e: (e.semantic_weight * max(e.fitts_id, 0.0)),
            reverse=True,
        )
        payload: list[dict[str, Any]] = []
        for e in scored[:max_items]:
            dx = e.center_x - origin[0]
            dy = e.center_y - origin[1]
            payload.append(
                {
                    "id": e.id,
                    "bbox": {"x": e.x, "y": e.y, "w": e.w, "h": e.h},
                    "role_guess": e.role_guess,
                    "semantic_weight": round(e.semantic_weight, 4),
                    "fitts_id": round(e.fitts_id, 4),
                    "distance_to_origin": round(math.hypot(dx, dy), 2),
                }
            )
        return payload

    def _normalize_brand_pattern_summary(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}

        patterns_raw = raw.get("design_patterns", [])
        if not isinstance(patterns_raw, list):
            patterns_raw = []

        patterns: list[dict[str, Any]] = []
        for item in patterns_raw:
            if not isinstance(item, dict):
                continue
            evidence = item.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            patterns.append(
                {
                    "name": str(item.get("name", "")).strip(),
                    "rule": str(item.get("rule", "")).strip(),
                    "evidence": [str(x).strip() for x in evidence if str(x).strip()],
                    "quant_signal": str(item.get("quant_signal", "")).strip(),
                }
            )

        baseline_raw = raw.get("score_baseline", {})
        if not isinstance(baseline_raw, dict):
            baseline_raw = {}
        consistency = str(baseline_raw.get("consistency", "medium")).lower().strip()
        if consistency not in {"high", "medium", "low"}:
            consistency = "medium"

        return {
            "brand_positioning": str(raw.get("brand_positioning", "")).strip(),
            "pattern_summary": str(raw.get("pattern_summary", "")).strip(),
            "design_patterns": patterns[:6],
            "score_baseline": {
                "balance_mean": self._safe_float(baseline_raw.get("balance_mean", 0.0)),
                "accessibility_mean": self._safe_float(baseline_raw.get("accessibility_mean", 0.0)),
                "final_mean": self._safe_float(baseline_raw.get("final_mean", 0.0)),
                "consistency": consistency,
            },
            "dos": [str(x).strip() for x in raw.get("dos", []) if str(x).strip()] if isinstance(raw.get("dos", []), list) else [],
            "donts": [str(x).strip() for x in raw.get("donts", []) if str(x).strip()] if isinstance(raw.get("donts", []), list) else [],
            "checklist": [str(x).strip() for x in raw.get("checklist", []) if str(x).strip()] if isinstance(raw.get("checklist", []), list) else [],
        }

    def _normalize_brand_gap_diagnosis(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {}

        alignment = str(raw.get("overall_alignment", "medium")).lower().strip()
        if alignment not in {"high", "medium", "low"}:
            alignment = "medium"

        score_alignment = raw.get("score_alignment", {})
        if not isinstance(score_alignment, dict):
            score_alignment = {}

        violations_raw = raw.get("violations", [])
        if not isinstance(violations_raw, list):
            violations_raw = []
        violations: list[dict[str, Any]] = []
        for item in violations_raw:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity", "medium")).lower().strip()
            if severity not in {"high", "medium", "low"}:
                severity = "medium"
            violations.append(
                {
                    "pattern": str(item.get("pattern", "")).strip(),
                    "issue": str(item.get("issue", "")).strip(),
                    "severity": severity,
                    "evidence": str(item.get("evidence", "")).strip(),
                }
            )

        improvements_raw = raw.get("improvements", [])
        if not isinstance(improvements_raw, list):
            improvements_raw = []
        improvements: list[dict[str, Any]] = []
        for item in improvements_raw:
            if not isinstance(item, dict):
                continue
            priority = str(item.get("priority", "P1")).upper().strip()
            if priority not in {"P0", "P1", "P2"}:
                priority = "P1"
            improvements.append(
                {
                    "priority": priority,
                    "title": str(item.get("title", "")).strip(),
                    "action": str(item.get("action", "")).strip(),
                    "expected_gain": str(item.get("expected_gain", "")).strip(),
                    "related_pattern": str(item.get("related_pattern", "")).strip(),
                }
            )

        quick_wins = raw.get("quick_wins", [])
        if not isinstance(quick_wins, list):
            quick_wins = []

        return {
            "overall_alignment": alignment,
            "summary": str(raw.get("summary", "")).strip(),
            "score_alignment": {
                "balance_gap": self._safe_float(score_alignment.get("balance_gap", 0.0)),
                "accessibility_gap": self._safe_float(score_alignment.get("accessibility_gap", 0.0)),
                "final_gap": self._safe_float(score_alignment.get("final_gap", 0.0)),
                "comment": str(score_alignment.get("comment", "")).strip(),
            },
            "violations": violations[:8],
            "improvements": improvements[:10],
            "quick_wins": [str(x).strip() for x in quick_wins if str(x).strip()][:8],
        }

    def _build_statistical_pattern_summary(
        self,
        brand_name: str,
        sample_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        balances: list[float] = []
        accesses: list[float] = []
        finals: list[float] = []
        for case in sample_cases:
            scores = case.get("scores", {}) if isinstance(case, dict) else {}
            if not isinstance(scores, dict):
                scores = {}
            balances.append(self._safe_float(scores.get("balance_10", 0.0)))
            accesses.append(self._safe_float(scores.get("accessibility_10", 0.0)))
            finals.append(self._safe_float(scores.get("final_10", 0.0)))

        if not balances:
            return {
                "brand_positioning": f"{brand_name or '该品牌'}页面特征不足，无法统计。",
                "pattern_summary": "样本数量不足，建议至少提供2-3张同品牌核心页面。",
                "design_patterns": [],
                "score_baseline": {
                    "balance_mean": 0.0,
                    "accessibility_mean": 0.0,
                    "final_mean": 0.0,
                    "consistency": "low",
                },
                "dos": [],
                "donts": [],
                "checklist": [],
            }

        balance_mean = float(np.mean(balances))
        access_mean = float(np.mean(accesses))
        final_mean = float(np.mean(finals))
        final_std = float(np.std(finals))

        if final_std <= 0.6:
            consistency = "high"
        elif final_std <= 1.2:
            consistency = "medium"
        else:
            consistency = "low"

        patterns: list[dict[str, Any]] = [
            {
                "name": "分数稳定区间",
                "rule": "新页面应优先保证总分接近品牌历史均值，并避免明显低于均值。",
                "evidence": [
                    f"样本均值 final={final_mean:.2f}/10，标准差={final_std:.2f}",
                ],
                "quant_signal": "最终得分偏离均值越大，越可能偏离品牌范式。",
            }
        ]

        if access_mean >= balance_mean:
            patterns.append(
                {
                    "name": "可达性优先倾向",
                    "rule": "高频核心操作优先放在拇指友好区，并保持足够点击尺寸。",
                    "evidence": [
                        f"可达性均值={access_mean:.2f}/10，高于或接近平衡均值={balance_mean:.2f}/10",
                    ],
                    "quant_signal": "可达性是品牌体验的一阶约束。",
                }
            )
        else:
            patterns.append(
                {
                    "name": "视觉秩序优先倾向",
                    "rule": "先稳定整体重心和信息层级，再优化交互路径。",
                    "evidence": [
                        f"平衡均值={balance_mean:.2f}/10，高于可达性均值={access_mean:.2f}/10",
                    ],
                    "quant_signal": "视觉平衡是品牌一致性的主要贡献项。",
                }
            )

        return {
            "brand_positioning": f"{brand_name or '该品牌'}在视觉平衡与交互可达性上有可量化的一致性特征。",
            "pattern_summary": "基于样本分数可提炼出稳定基线，可用于后续页面一致性评审。",
            "design_patterns": patterns,
            "score_baseline": {
                "balance_mean": round(balance_mean, 3),
                "accessibility_mean": round(access_mean, 3),
                "final_mean": round(final_mean, 3),
                "consistency": consistency,
            },
            "dos": [
                "让关键操作入口优先落在屏幕中下部可达区域。",
                "控制视觉重心偏移，避免头重脚轻。",
                "保证高优先级按钮的点击尺寸。",
            ],
            "donts": [
                "避免把唯一高频入口放在右上角且尺寸过小。",
                "避免大面积高对比元素长期集中在上半屏。",
            ],
            "checklist": [
                "新页面 final 分数与品牌均值差值是否在1.0分以内。",
                "核心CTA是否位于拇指友好区并满足最小点击尺寸。",
                "视觉重心是否接近几何中心并与品牌样本趋势一致。",
            ],
        }

    def _build_heuristic_gap_diagnosis(
        self,
        *,
        brand_name: str,
        brand_pattern_summary: dict[str, Any],
        candidate_case: dict[str, Any],
    ) -> dict[str, Any]:
        baseline = brand_pattern_summary.get("score_baseline", {}) if isinstance(brand_pattern_summary, dict) else {}
        if not isinstance(baseline, dict):
            baseline = {}

        scores = candidate_case.get("scores", {}) if isinstance(candidate_case, dict) else {}
        if not isinstance(scores, dict):
            scores = {}

        balance_gap = self._safe_float(scores.get("balance_10", 0.0)) - self._safe_float(baseline.get("balance_mean", 0.0))
        access_gap = self._safe_float(scores.get("accessibility_10", 0.0)) - self._safe_float(baseline.get("accessibility_mean", 0.0))
        final_gap = self._safe_float(scores.get("final_10", 0.0)) - self._safe_float(baseline.get("final_mean", 0.0))

        if final_gap >= -0.5:
            alignment = "high"
        elif final_gap >= -1.5:
            alignment = "medium"
        else:
            alignment = "low"

        violations: list[dict[str, Any]] = []
        if balance_gap < -0.8:
            violations.append(
                {
                    "pattern": "视觉重心稳定",
                    "issue": "当前页面视觉平衡低于品牌基线，重心可能偏移明显。",
                    "severity": "high" if balance_gap < -1.4 else "medium",
                    "evidence": f"balance gap={balance_gap:.2f}",
                }
            )
        if access_gap < -0.8:
            violations.append(
                {
                    "pattern": "核心操作可达",
                    "issue": "当前页面可达性低于品牌基线，核心操作触达成本偏高。",
                    "severity": "high" if access_gap < -1.4 else "medium",
                    "evidence": f"accessibility gap={access_gap:.2f}",
                }
            )

        improvements: list[dict[str, Any]] = []
        if access_gap < 0:
            improvements.append(
                {
                    "priority": "P0",
                    "title": "下移核心CTA并放大点击区",
                    "action": "将最关键转化按钮下移至屏幕下35%范围，并将按钮高度提升到44-56dp。",
                    "expected_gain": "降低费茨成本，优先修复可达性短板。",
                    "related_pattern": "核心操作可达",
                }
            )
        if balance_gap < 0:
            improvements.append(
                {
                    "priority": "P1",
                    "title": "增加底部视觉锚点",
                    "action": "在底部增加稳定高对比操作条，分散上半屏视觉负荷。",
                    "expected_gain": "重心回归中心附近，提升视觉稳定感。",
                    "related_pattern": "视觉重心稳定",
                }
            )
        improvements.append(
            {
                "priority": "P2",
                "title": "建立品牌页签检查清单",
                "action": "按品牌范式 checklist 在评审环节逐条核对。",
                "expected_gain": "减少后续页面偏离品牌范式的概率。",
                "related_pattern": "流程规范",
            }
        )

        return {
            "overall_alignment": alignment,
            "summary": (
                f"候选页面相对{brand_name or '品牌'}基线的总分差值为 {final_gap:.2f}。"
                "建议优先修复可达性与视觉重心的主要差距。"
            ),
            "score_alignment": {
                "balance_gap": round(balance_gap, 3),
                "accessibility_gap": round(access_gap, 3),
                "final_gap": round(final_gap, 3),
                "comment": "负值表示低于品牌均值，且绝对值越大偏离越明显。",
            },
            "violations": violations,
            "improvements": improvements,
            "quick_wins": [
                "优先处理页面唯一核心按钮的位置和尺寸。",
                "将上半屏最重视觉块做减重或向中轴回收。",
            ],
        }

    def _safe_float(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _normalize_diagnosis(self, raw: dict[str, Any]) -> dict[str, Any]:
        def _normalize_dimension(data: Any) -> dict[str, Any]:
            if not isinstance(data, dict):
                return {
                    "level": "medium",
                    "why": "",
                    "rule_explanation": "",
                    "evidence": [],
                }

            evidence = data.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []

            level = str(data.get("level", "medium")).lower().strip()
            if level not in {"high", "medium", "low"}:
                level = "medium"

            return {
                "level": level,
                "why": str(data.get("why", "")).strip(),
                "rule_explanation": str(data.get("rule_explanation", "")).strip(),
                "evidence": [str(item).strip() for item in evidence if str(item).strip()],
            }

        plan_raw = raw.get("optimization_plan", [])
        if not isinstance(plan_raw, list):
            plan_raw = []

        normalized_plan: list[dict[str, Any]] = []
        for item in plan_raw:
            if not isinstance(item, dict):
                continue
            priority = str(item.get("priority", "P1")).upper().strip()
            if priority not in {"P0", "P1", "P2"}:
                priority = "P1"

            target_elements = item.get("target_elements", [])
            if not isinstance(target_elements, list):
                target_elements = []

            actions = item.get("actions", [])
            if not isinstance(actions, list):
                actions = []

            normalized_plan.append(
                {
                    "priority": priority,
                    "title": str(item.get("title", "")).strip(),
                    "target_elements": [int(x) for x in target_elements if isinstance(x, int)],
                    "problem": str(item.get("problem", "")).strip(),
                    "design_rule": str(item.get("design_rule", "")).strip(),
                    "actions": [str(step).strip() for step in actions if str(step).strip()],
                    "expected_effect": str(item.get("expected_effect", "")).strip(),
                    "verification": str(item.get("verification", "")).strip(),
                }
            )

        return {
            "overall_summary": str(raw.get("overall_summary", "")).strip(),
            "visual_balance": _normalize_dimension(raw.get("visual_balance", {})),
            "interaction_accessibility": _normalize_dimension(raw.get("interaction_accessibility", {})),
            "optimization_plan": normalized_plan,
        }

    def _query_llm(self, payload: list[dict[str, Any]], image_source: str | Path | np.ndarray) -> dict[str, Any]:
        image_b64 = self._encode_image(image_source)
        prompt = (
            "You are an expert mobile UX evaluator."
            "\nGiven UI screenshot and component list, assign semantic priority alpha [0.1, 1.0]."
            "\nHigh-frequency core CTA should be near 1.0, secondary actions around 0.5-0.8,"
            " low-frequency controls around 0.1-0.4."
            "\nReturn STRICT JSON: {\"weights\":[{\"id\":1,\"alpha\":0.9,\"reason\":\"...\"}],\"notes\":\"...\"}"
            f"\ncomponents={json.dumps(payload, ensure_ascii=False)}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        content = response.choices[0].message.content
        return json.loads(content)

    def _merge_weights(self, elements: list[UIElement], response: dict[str, Any]) -> None:
        mapping = {int(item["id"]): item for item in response.get("weights", []) if "id" in item}
        for e in elements:
            item = mapping.get(e.id)
            if not item:
                continue
            alpha = float(item.get("alpha", 0.5))
            e.semantic_weight = max(0.1, min(1.0, alpha))

    def _encode_image(self, image_source: str | Path | np.ndarray) -> str:
        if isinstance(image_source, np.ndarray):
            if cv2 is None:
                raise RuntimeError("cv2 is required to encode ndarray image for LLM")
            ok, buffer = cv2.imencode(".png", image_source)
            if not ok:
                raise ValueError("Failed to encode ndarray image")
            return base64.b64encode(buffer.tobytes()).decode("utf-8")

        with open(image_source, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
