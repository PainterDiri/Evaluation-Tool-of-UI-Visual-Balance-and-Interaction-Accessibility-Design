from __future__ import annotations

import math
from pathlib import Path

from .config import RuntimeConfig, score_0_to_10
from .guideline_manager import GuidelineManager
from .image_analyzer import ImageAnalyzer
from .metrics import compute_accessibility, compute_physical_balance
from .reporting import save_json_report
from .semantic_weighting import SemanticWeighter
from .ui_types import EvaluationResult, ScoreBundle
from .visualization import save_visuals


class EvaluationPipeline:
    def __init__(self, config: RuntimeConfig, guideline_file: str | Path = "data/guidelines.json"):
        self.config = config
        self.config.validate()
        self.guidelines = GuidelineManager(guideline_file)
        self.analyzer = ImageAnalyzer(
            min_area_ratio=config.min_area_ratio,
            max_elements=config.max_elements,
            top_crop_ratio=config.top_crop_ratio,
        )
        self.weighter = SemanticWeighter(
            use_llm=config.use_llm,
            model=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )

    def run(self, image_path: str | Path, save_artifacts: bool = True) -> dict:
        analysis = self.analyzer.analyze(image_path)
        elements = analysis.elements

        semantic_meta = self.weighter.apply(elements, analysis.image)

        physical_balance, center_of_mass, geometric_center = compute_physical_balance(
            elements=elements,
            width=analysis.width,
            height=analysis.height,
        )

        interactive = [e for e in elements if e.is_interactive]
        accessibility, weighted_cost, origin = compute_accessibility(
            interactive_elements=interactive,
            width=analysis.width,
            height=analysis.height,
            origin_mode=self.config.origin_mode,
            max_reasonable_id=self.config.accessibility_max_reasonable_id,
        )

        base_recommendations = self._build_recommendations(
            physical_balance=physical_balance,
            accessibility=accessibility,
            weighted_cost=weighted_cost,
            interactive=interactive,
            width=analysis.width,
            height=analysis.height,
            center_of_mass=center_of_mass,
            geometric_center=geometric_center,
        )

        detailed_meta = self.weighter.generate_detailed_diagnosis(
            image_source=analysis.image,
            width=analysis.width,
            height=analysis.height,
            physical_balance=physical_balance,
            accessibility=accessibility,
            weighted_cost=weighted_cost,
            center_of_mass=center_of_mass,
            geometric_center=geometric_center,
            origin=origin,
            interactive_elements=interactive,
            base_recommendations=base_recommendations,
        )

        recommendations = self._recommendations_from_detailed(
            detailed_meta=detailed_meta,
            base_recommendations=base_recommendations,
        )

        final_0_to_1 = (
            self.config.balance_weight * physical_balance
            + self.config.accessibility_weight * accessibility
        )

        result = EvaluationResult(
            image_path=str(image_path),
            width=analysis.width,
            height=analysis.height,
            elements_total=len(elements),
            interactive_total=len(interactive),
            center_of_mass=center_of_mass,
            geometric_center=geometric_center,
            score_bundle=ScoreBundle(
                physical_balance=physical_balance,
                accessibility=accessibility,
                weighted_interaction_cost=weighted_cost,
                final_score=final_0_to_1,
            ),
            recommendations=recommendations,
            metadata={
                "origin": {"x": round(origin[0], 2), "y": round(origin[1], 2)},
                "preprocess": {
                    "top_crop_px": analysis.crop_top_px,
                    "top_crop_ratio": round(
                        analysis.crop_top_px / max(analysis.original_height, 1),
                        4,
                    ),
                    "original_size": {
                        "width": analysis.original_width,
                        "height": analysis.original_height,
                    },
                    "effective_size": {
                        "width": analysis.width,
                        "height": analysis.height,
                    },
                },
                "semantic_mode": semantic_meta,
                "detailed_analysis_mode": detailed_meta.get("mode", "none"),
                "detailed_analysis_reason": detailed_meta.get("reason", ""),
                "detailed_analysis": detailed_meta.get("diagnosis", {}),
                "guidelines_loaded": len(self.guidelines.guidelines),
                "score_10": {
                    "physical_balance": score_0_to_10(physical_balance),
                    "accessibility": score_0_to_10(accessibility),
                    "final": score_0_to_10(final_0_to_1),
                },
            },
            elements=elements,
        )

        artifact_paths: dict[str, str] = {}
        report_path = ""
        if save_artifacts:
            artifact_paths = save_visuals(result, analysis.image, self.config.output_dir, self.config.origin_mode)
            report_path = save_json_report(result, self.config.output_dir)

        return {
            "result": result,
            "report_path": report_path,
            "artifact_paths": artifact_paths,
        }

    def _build_recommendations(
        self,
        physical_balance: float,
        accessibility: float,
        weighted_cost: float,
        interactive,
        width: int,
        height: int,
        center_of_mass: tuple[float, float],
        geometric_center: tuple[float, float],
    ) -> list[str]:
        recs: list[str] = []

        dx = center_of_mass[0] - geometric_center[0]
        dy = center_of_mass[1] - geometric_center[1]
        max_dist = math.hypot(geometric_center[0], geometric_center[1])
        offset_ratio = 0.0 if max_dist <= 1e-9 else math.hypot(dx, dy) / max_dist

        balance_level = self._score_level(physical_balance, high=0.82, medium=0.68)
        access_level = self._score_level(accessibility, high=0.75, medium=0.58)

        recs.append(
            "[视觉平衡诊断] "
            f"得分 {physical_balance:.3f}（{self._level_to_cn(balance_level)}），"
            f"视觉重心相对几何中心偏移 dx={dx:.1f}px, dy={dy:.1f}px, 归一化偏移={offset_ratio:.3f}。"
        )

        recs.append(
            "[交互可达性诊断] "
            f"得分 {accessibility:.3f}（{self._level_to_cn(access_level)}），"
            f"语义加权费茨成本={weighted_cost:.3f}（越低越好）。"
        )

        if physical_balance < 0.75:
            recs.append(
                "[视觉平衡优化] 当前画面存在重心偏移，建议按优先级处理："
                "1) 在屏幕下半区增加稳定锚点（底部固定操作条/结算条）；"
                "2) 将高对比主按钮放在中下区域，减少顶部视觉负担；"
                "3) 对顶部大面积高对比区进行降噪（降低饱和度、减小面积或拆分模块）。"
            )
        elif physical_balance < 0.85:
            recs.append(
                "[视觉平衡优化] 当前平衡性中等，建议做精修："
                "将最重的1-2个视觉块向屏幕中轴靠拢，并在对侧补充等量视觉重量，"
                "使重心偏移进一步收敛。"
            )

        if accessibility < 0.65:
            recs.append(
                "[交互可达性优化] 核心操作触达成本偏高，建议按顺序执行："
                "1) 将核心 CTA 下移到屏幕下35%区域；"
                "2) 将关键点击目标高度提升到44-56dp，宽度尽量接近容器宽度；"
                "3) 对高频操作提供底部冗余入口，避免仅保留顶部入口。"
            )
        elif accessibility < 0.78:
            recs.append(
                "[交互可达性优化] 可达性中等，可进一步降低成本："
                "优先放大高频按钮短边尺寸，随后再做小幅下移，以较小改动获得稳定收益。"
            )

        high_priority_hard = []
        for e in interactive:
            if e.semantic_weight >= 0.75 and e.fitts_id >= 2.6:
                high_priority_hard.append((e.id, e.fitts_id, e.semantic_weight))

        if high_priority_hard:
            hard_text = ", ".join(
                [f"#{eid}(ID={fid:.2f}, alpha={alpha:.2f})" for eid, fid, alpha in high_priority_hard[:8]]
            )
            recs.append(
                "[高优先级风险点] 以下组件语义重要但点击成本偏高："
                f"{hard_text}。建议优先改造这些组件的位置与尺寸。"
            )

        if not recs:
            recs.append("当前界面在视觉平衡与交互可达性上整体表现良好，可重点优化内容密度与信息层级细节。")

        return recs

    def _recommendations_from_detailed(
        self,
        detailed_meta: dict,
        base_recommendations: list[str],
    ) -> list[str]:
        if detailed_meta.get("mode") != "llm-detailed":
            return base_recommendations

        diagnosis = detailed_meta.get("diagnosis", {})
        if not isinstance(diagnosis, dict):
            return base_recommendations

        recs: list[str] = []
        summary = str(diagnosis.get("overall_summary", "")).strip()
        if summary:
            recs.append(f"[总体结论] {summary}")

        for key, title in [
            ("visual_balance", "视觉平衡"),
            ("interaction_accessibility", "交互可达性"),
        ]:
            dim = diagnosis.get(key, {})
            if not isinstance(dim, dict):
                continue

            level = self._level_to_cn(str(dim.get("level", "medium")))
            why = str(dim.get("why", "")).strip()
            rule = str(dim.get("rule_explanation", "")).strip()
            evidence = dim.get("evidence", [])
            if not isinstance(evidence, list):
                evidence = []
            evidence_text = "；".join([str(x).strip() for x in evidence if str(x).strip()][:4])

            text = f"[{title}解释] 等级={level}。"
            if why:
                text += f"原因：{why}。"
            if evidence_text:
                text += f"证据：{evidence_text}。"
            if rule:
                text += f"规则依据：{rule}。"
            recs.append(text)

        plan = diagnosis.get("optimization_plan", [])
        if isinstance(plan, list):
            for item in plan[:6]:
                if not isinstance(item, dict):
                    continue
                priority = str(item.get("priority", "P1")).strip().upper() or "P1"
                title = str(item.get("title", "未命名优化项")).strip()
                targets = item.get("target_elements", [])
                if not isinstance(targets, list):
                    targets = []
                target_text = ", ".join([f"#{x}" for x in targets if isinstance(x, int)])
                problem = str(item.get("problem", "")).strip()
                rule = str(item.get("design_rule", "")).strip()
                actions = item.get("actions", [])
                if not isinstance(actions, list):
                    actions = []
                action_text = " -> ".join([str(x).strip() for x in actions if str(x).strip()][:5])
                effect = str(item.get("expected_effect", "")).strip()
                verify = str(item.get("verification", "")).strip()

                text = f"[优化{priority}] {title}。"
                if target_text:
                    text += f"目标组件：{target_text}。"
                if problem:
                    text += f"问题：{problem}。"
                if rule:
                    text += f"依据：{rule}。"
                if action_text:
                    text += f"执行步骤：{action_text}。"
                if effect:
                    text += f"预期效果：{effect}。"
                if verify:
                    text += f"验证方式：{verify}。"
                recs.append(text)

        return recs if recs else base_recommendations

    def _score_level(self, score: float, high: float, medium: float) -> str:
        if score >= high:
            return "high"
        if score >= medium:
            return "medium"
        return "low"

    def _level_to_cn(self, level: str) -> str:
        normalized = level.lower().strip()
        if normalized == "high":
            return "高"
        if normalized == "low":
            return "低"
        return "中"
