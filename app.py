from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.batch_analysis import run_batch_analysis
from src.config import RuntimeConfig
from src.pipeline import EvaluationPipeline
from src.paths import INPUT_DIR, OUTPUT_DIR, list_input_images


st.set_page_config(
    page_title="UI Visual Balance & Accessibility Evaluator",
    page_icon="📱",
    layout="wide",
)

st.title("移动端 UI 视觉平衡与交互可达性评估")
st.caption("Hybrid Pipeline: OpenCV 几何计算 + 可选 MLLM 语义赋权")


def _materialize_image_input(image_obj: Any) -> Path:
    if isinstance(image_obj, Path):
        return image_obj

    name = str(getattr(image_obj, "name", "uploaded.png"))
    suffix = Path(name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_obj.getbuffer())
        return Path(tmp.name)


def _build_case_payload(image_name: str, result: Any) -> dict[str, Any]:
    score_10 = result.metadata.get("score_10", {}) if isinstance(result.metadata, dict) else {}
    if not isinstance(score_10, dict):
        score_10 = {}

    interactive = [e for e in result.elements if e.is_interactive]
    high_risk = sorted(
        interactive,
        key=lambda e: (float(e.semantic_weight) * max(float(e.fitts_id), 0.0)),
        reverse=True,
    )

    detail = result.metadata.get("detailed_analysis", {}) if isinstance(result.metadata, dict) else {}
    if not isinstance(detail, dict):
        detail = {}

    return {
        "image_name": image_name,
        "scores": {
            "balance_10": float(score_10.get("physical_balance", 0.0)),
            "accessibility_10": float(score_10.get("accessibility", 0.0)),
            "final_10": float(score_10.get("final", 0.0)),
            "balance_01": float(result.score_bundle.physical_balance),
            "accessibility_01": float(result.score_bundle.accessibility),
            "final_01": float(result.score_bundle.final_score),
        },
        "metrics": {
            "weighted_interaction_cost": float(result.score_bundle.weighted_interaction_cost),
            "elements_total": int(result.elements_total),
            "interactive_total": int(result.interactive_total),
            "center_of_mass": {
                "x": round(float(result.center_of_mass[0]), 2),
                "y": round(float(result.center_of_mass[1]), 2),
            },
            "geometric_center": {
                "x": round(float(result.geometric_center[0]), 2),
                "y": round(float(result.geometric_center[1]), 2),
            },
        },
        "top_risk_elements": [
            {
                "id": int(e.id),
                "role_guess": str(e.role_guess),
                "semantic_weight": round(float(e.semantic_weight), 4),
                "fitts_id": round(float(e.fitts_id), 4),
                "bbox": {
                    "x": int(e.x),
                    "y": int(e.y),
                    "w": int(e.w),
                    "h": int(e.h),
                },
            }
            for e in high_risk[:8]
        ],
        "recommendations": [str(x) for x in result.recommendations[:5]],
        "detailed_snapshot": {
            "overall_summary": str(detail.get("overall_summary", "")).strip(),
            "visual_balance_level": str(detail.get("visual_balance", {}).get("level", "")) if isinstance(detail.get("visual_balance", {}), dict) else "",
            "interaction_accessibility_level": str(detail.get("interaction_accessibility", {}).get("level", "")) if isinstance(detail.get("interaction_accessibility", {}), dict) else "",
        },
    }

with st.sidebar:
    st.header("运行参数")
    origin_mode = st.selectbox("拇指起点", ["bottom-center", "bottom-right"], index=0)
    use_llm = st.toggle("启用 LLM 语义赋权", value=False)
    min_area_ratio = st.slider("最小组件面积比例", min_value=0.0005, max_value=0.008, value=0.0015, step=0.0005)
    max_elements = st.slider("最大组件数量", min_value=60, max_value=300, value=180, step=10)
    top_crop_ratio = st.slider("顶部裁剪比例", min_value=0.0, max_value=0.12, value=0.04, step=0.005)
    accessibility_max_id = st.slider("可达性标定上限ID", min_value=4.0, max_value=8.0, value=6.2, step=0.1)
    st.header("批量象限参数")
    batch_balance_threshold = st.slider("平衡度分割阈值", min_value=0.50, max_value=0.95, value=0.75, step=0.01)
    batch_access_threshold = st.slider("可达性分割阈值", min_value=0.50, max_value=0.95, value=0.65, step=0.01)

default_images = list_input_images()
uploaded = st.file_uploader("上传 UI 截图", type=["png", "jpg", "jpeg", "webp"])

st.caption(f"默认输入目录：{INPUT_DIR}")

source_choice = st.radio(
    "选择输入来源",
    ["使用 input/ 目录中的默认图片", "上传图片"],
    index=0,
    horizontal=True,
)

selected_image = None
selected_name = ""

if source_choice == "上传图片":
    selected_image = uploaded
    selected_name = uploaded.name if uploaded is not None else ""
    if uploaded is None:
        st.info("请先上传一张图片，或者切换回默认图片模式。")
else:
    if default_images:
        default_names = [path.name for path in default_images]
        selected_name = st.selectbox("选择默认图片", default_names, index=0)
        selected_image = default_images[default_names.index(selected_name)]
    else:
        st.warning("input/ 目录下没有找到图片，请先放入 png/jpg/jpeg/webp 文件。")

if selected_image is not None:
    cols = st.columns([1.1, 1.2])
    with cols[0]:
        preview = selected_image if not isinstance(selected_image, Path) else str(selected_image)
        st.image(preview, caption=f"输入截图：{selected_name or Path(selected_image).name}", use_container_width=True)

    if st.button("开始评估", type="primary"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if isinstance(selected_image, Path):
            temp_path = selected_image
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(selected_image.name).suffix) as tmp:
                tmp.write(selected_image.getbuffer())
                temp_path = Path(tmp.name)

        config = RuntimeConfig(
            origin_mode=origin_mode,
            use_llm=use_llm,
            min_area_ratio=min_area_ratio,
            max_elements=max_elements,
            top_crop_ratio=top_crop_ratio,
            accessibility_max_reasonable_id=accessibility_max_id,
            output_dir=OUTPUT_DIR,
        )

        with st.spinner("正在执行评估..."):
            pipeline = EvaluationPipeline(config=config)
            output = pipeline.run(image_path=temp_path, save_artifacts=True)
            result = output["result"]

        score = result.metadata["score_10"]
        st.success("评估完成")

        c1, c2, c3 = st.columns(3)
        c1.metric("视觉平衡(0-10)", score["physical_balance"])
        c2.metric("可达性(0-10)", score["accessibility"])
        c3.metric("总分(0-10)", score["final"])

        with cols[1]:
            st.subheader("核心指标")
            st.write(
                {
                    "elements_total": result.elements_total,
                    "interactive_total": result.interactive_total,
                    "weighted_cost": round(result.score_bundle.weighted_interaction_cost, 4),
                    "semantic_mode": result.metadata.get("semantic_mode", {}),
                    "detailed_analysis_mode": result.metadata.get("detailed_analysis_mode", "none"),
                }
            )
            st.subheader("优化建议")
            for i, rec in enumerate(result.recommendations, start=1):
                st.write(f"{i}. {rec}")

        detail_mode = result.metadata.get("detailed_analysis_mode", "none")
        detail_reason = result.metadata.get("detailed_analysis_reason", "")
        detail = result.metadata.get("detailed_analysis", {})

        if isinstance(detail, dict) and detail:
            st.subheader("LLM 结构化诊断")
            st.caption(f"模式：{detail_mode}")
            if detail.get("overall_summary"):
                st.info(str(detail["overall_summary"]))

            diag_cols = st.columns(2)
            with diag_cols[0]:
                st.markdown("**视觉平衡性：为什么高/低**")
                vb = detail.get("visual_balance", {}) if isinstance(detail.get("visual_balance", {}), dict) else {}
                st.write(f"等级：{vb.get('level', 'medium')}")
                if vb.get("why"):
                    st.write(f"原因：{vb['why']}")
                if vb.get("rule_explanation"):
                    st.write(f"规则解释：{vb['rule_explanation']}")
                evidence = vb.get("evidence", []) if isinstance(vb.get("evidence", []), list) else []
                if evidence:
                    st.write("证据：")
                    for idx, item in enumerate(evidence[:6], start=1):
                        st.write(f"{idx}. {item}")

            with diag_cols[1]:
                st.markdown("**交互可达性：为什么高/低**")
                ia = detail.get("interaction_accessibility", {}) if isinstance(detail.get("interaction_accessibility", {}), dict) else {}
                st.write(f"等级：{ia.get('level', 'medium')}")
                if ia.get("why"):
                    st.write(f"原因：{ia['why']}")
                if ia.get("rule_explanation"):
                    st.write(f"规则解释：{ia['rule_explanation']}")
                evidence = ia.get("evidence", []) if isinstance(ia.get("evidence", []), list) else []
                if evidence:
                    st.write("证据：")
                    for idx, item in enumerate(evidence[:6], start=1):
                        st.write(f"{idx}. {item}")

            plan = detail.get("optimization_plan", []) if isinstance(detail.get("optimization_plan", []), list) else []
            if plan:
                st.subheader("LLM 优化计划（按优先级）")
                for idx, item in enumerate(plan[:8], start=1):
                    if not isinstance(item, dict):
                        continue
                    title = str(item.get("title", f"优化项 {idx}")).strip() or f"优化项 {idx}"
                    priority = str(item.get("priority", "P1")).strip().upper() or "P1"
                    with st.expander(f"{idx}. [{priority}] {title}", expanded=(idx <= 2)):
                        targets = item.get("target_elements", [])
                        if isinstance(targets, list) and targets:
                            st.write(f"目标组件：{', '.join([f'#{t}' for t in targets if isinstance(t, int)])}")
                        if item.get("problem"):
                            st.write(f"问题：{item['problem']}")
                        if item.get("design_rule"):
                            st.write(f"设计规则：{item['design_rule']}")
                        actions = item.get("actions", []) if isinstance(item.get("actions", []), list) else []
                        if actions:
                            st.write("执行步骤：")
                            for step_idx, step in enumerate(actions[:8], start=1):
                                st.write(f"{step_idx}. {step}")
                        if item.get("expected_effect"):
                            st.write(f"预期效果：{item['expected_effect']}")
                        if item.get("verification"):
                            st.write(f"验证方式：{item['verification']}")
        elif detail_mode == "fallback" and detail_reason:
            st.warning(f"LLM 详细诊断调用失败，已回退到规则建议。原因：{detail_reason}")

        st.subheader("可视化结果")
        visual_cols = st.columns(2)
        artifact_paths = output["artifact_paths"]
        with visual_cols[0]:
            st.image(artifact_paths["elements_annotated"], caption="组件识别与语义权重", use_container_width=True)
            st.image(artifact_paths["visual_balance_overlay"], caption="视觉重心与几何中心", use_container_width=True)
        with visual_cols[1]:
            st.image(artifact_paths["reachability_heatmap"], caption="可达性热力图", use_container_width=True)
            st.image(artifact_paths["score_radar"], caption="维度雷达图", use_container_width=True)

        st.subheader("JSON 报告")
        st.code(output["report_path"], language="text")
        with open(output["report_path"], "r", encoding="utf-8") as f:
            st.download_button("下载报告", f.read(), file_name="evaluation_report.json", mime="application/json")
else:
    st.info("请上传一张移动端 UI 截图，或把图片放到 input/ 目录后使用默认图片模式。")

st.divider()
st.subheader("批量评估：input/ 象限分布")
st.caption("评估 input/ 目录中的全部样本，并将每张图以编号映射到象限图中。")

batch_images = list_input_images()
if not batch_images:
    st.warning("input/ 目录中暂无可用于批量评估的图片。")
else:
    st.write(f"当前可用于批量评估的图片数量：{len(batch_images)}")

    if st.button("开始批量评估并绘制象限图", type="secondary"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        config = RuntimeConfig(
            origin_mode=origin_mode,
            use_llm=use_llm,
            min_area_ratio=min_area_ratio,
            max_elements=max_elements,
            top_crop_ratio=top_crop_ratio,
            accessibility_max_reasonable_id=accessibility_max_id,
            output_dir=OUTPUT_DIR,
        )
        pipeline = EvaluationPipeline(config=config)

        progress_bar = st.progress(0.0, text="准备开始批量评估...")
        status_line = st.empty()

        def _on_progress(index: int, total: int, image_path: Path) -> None:
            progress_bar.progress(index / total, text=f"处理中 {index}/{total}")
            status_line.caption(f"当前样本：{image_path.name}")

        with st.spinner("正在批量评估并生成象限图..."):
            batch_output = run_batch_analysis(
                pipeline=pipeline,
                image_paths=batch_images,
                output_dir=OUTPUT_DIR,
                save_individual_artifacts=False,
                x_threshold=batch_balance_threshold,
                y_threshold=batch_access_threshold,
                progress_callback=_on_progress,
            )

        progress_bar.empty()
        status_line.empty()
        st.success("批量评估完成")

        q = batch_output["quadrant_counts"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Q1", q.get("Q1", 0))
        c2.metric("Q2", q.get("Q2", 0))
        c3.metric("Q3", q.get("Q3", 0))
        c4.metric("Q4", q.get("Q4", 0))

        st.image(batch_output["quadrant_plot"], caption="象限图（编号点位 + 图片映射）", use_container_width=True)
        st.image(batch_output["quadrant_plot_with_thumbnails"], caption="象限图（缩略图叠加）", use_container_width=True)

        st.subheader("编号与样本对应")
        table_rows = [
            {
                "index": row["index"],
                "image_name": row["image_name"],
                "quadrant": row["quadrant"],
                "balance": round(row["physical_balance"], 4),
                "accessibility": round(row["accessibility"], 4),
                "final_score": round(row["final_score"], 4),
            }
            for row in batch_output["records"]
        ]
        st.dataframe(table_rows, hide_index=True, use_container_width=True)

        st.subheader("批量结果 JSON")
        st.code(batch_output["summary_path"], language="text")
        with open(batch_output["summary_path"], "r", encoding="utf-8") as f:
            st.download_button(
                "下载批量报告",
                data=f.read(),
                file_name="batch_evaluation_summary.json",
                mime="application/json",
            )

st.divider()
st.subheader("品牌范式提炼与新图对照诊断")
st.caption("选择同品牌多张样本图提炼范式，再对另一张候选图做评分并进行差距诊断。")

brand_name = st.text_input("品牌名称", value="美团", key="brand_name_input")

brand_sample_source = st.radio(
    "品牌样本来源",
    ["从 input/ 目录多选", "上传多张品牌样本图"],
    horizontal=True,
    key="brand_sample_source",
)

brand_sample_items: list[tuple[str, Any]] = []
if brand_sample_source == "从 input/ 目录多选":
    input_names = [path.name for path in default_images]
    default_selection = input_names[: min(3, len(input_names))]
    selected_brand_names = st.multiselect(
        "选择同品牌样本图（建议 3-8 张）",
        input_names,
        default=default_selection,
        key="brand_sample_select",
    )
    name_to_path = {path.name: path for path in default_images}
    brand_sample_items = [(name, name_to_path[name]) for name in selected_brand_names if name in name_to_path]
else:
    uploaded_brand_samples = st.file_uploader(
        "上传同品牌样本图（可多选）",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="brand_sample_upload",
    )
    brand_sample_items = [
        (str(file.name), file)
        for file in (uploaded_brand_samples or [])
    ]

candidate_source = st.radio(
    "候选对照图来源",
    ["从 input/ 目录选择", "上传候选图"],
    horizontal=True,
    key="candidate_source",
)

candidate_item: tuple[str, Any] | None = None
if candidate_source == "从 input/ 目录选择":
    input_names = [path.name for path in default_images]
    if input_names:
        candidate_name = st.selectbox("选择候选图", input_names, key="candidate_select")
        name_to_path = {path.name: path for path in default_images}
        if candidate_name in name_to_path:
            candidate_item = (candidate_name, name_to_path[candidate_name])
    else:
        st.warning("input/ 目录没有可选图片，请上传候选图。")
else:
    uploaded_candidate = st.file_uploader(
        "上传候选图",
        type=["png", "jpg", "jpeg", "webp"],
        key="candidate_upload",
    )
    if uploaded_candidate is not None:
        candidate_item = (str(uploaded_candidate.name), uploaded_candidate)

if st.button("生成品牌范式并诊断候选图", type="primary", key="run_brand_pattern"):
    if len(brand_sample_items) < 2:
        st.error("请至少提供 2 张同品牌样本图。")
    elif candidate_item is None:
        st.error("请提供 1 张候选对照图。")
    else:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config = RuntimeConfig(
            origin_mode=origin_mode,
            use_llm=use_llm,
            min_area_ratio=min_area_ratio,
            max_elements=max_elements,
            top_crop_ratio=top_crop_ratio,
            accessibility_max_reasonable_id=accessibility_max_id,
            output_dir=OUTPUT_DIR,
        )

        if not use_llm:
            st.warning("当前未启用 LLM，范式提炼与对照诊断将自动回退为统计规则模式。")

        with st.spinner("正在进行品牌范式提炼与候选图诊断..."):
            try:
                pipeline = EvaluationPipeline(config=config)
                total_steps = len(brand_sample_items) + 2
                step = 0
                progress_bar = st.progress(0.0, text="开始处理...")

                sample_cases: list[dict[str, Any]] = []
                sample_images: list[Path] = []
                for image_name, image_obj in brand_sample_items:
                    image_path = _materialize_image_input(image_obj)
                    sample_images.append(image_path)

                    output = pipeline.run(image_path=image_path, save_artifacts=False)
                    sample_cases.append(_build_case_payload(image_name=image_name, result=output["result"]))

                    step += 1
                    progress_bar.progress(step / total_steps, text=f"样本评估中 {step}/{total_steps}")

                brand_meta = pipeline.weighter.summarize_brand_pattern(
                    brand_name=brand_name,
                    sample_cases=sample_cases,
                    sample_images=sample_images,
                )
                brand_summary = brand_meta.get("summary", {}) if isinstance(brand_meta, dict) else {}
                if not isinstance(brand_summary, dict):
                    brand_summary = {}

                step += 1
                progress_bar.progress(step / total_steps, text=f"范式提炼完成 {step}/{total_steps}")

                candidate_name, candidate_obj = candidate_item
                candidate_path = _materialize_image_input(candidate_obj)
                candidate_output = pipeline.run(image_path=candidate_path, save_artifacts=True)
                candidate_result = candidate_output["result"]
                candidate_case = _build_case_payload(image_name=candidate_name, result=candidate_result)

                diagnosis_meta = pipeline.weighter.diagnose_against_brand_pattern(
                    brand_name=brand_name,
                    brand_pattern_summary=brand_summary,
                    candidate_case=candidate_case,
                    candidate_image=candidate_path,
                )
                diagnosis = diagnosis_meta.get("diagnosis", {}) if isinstance(diagnosis_meta, dict) else {}
                if not isinstance(diagnosis, dict):
                    diagnosis = {}

                step += 1
                progress_bar.progress(1.0, text=f"候选图诊断完成 {step}/{total_steps}")
            except Exception as exc:
                st.error(f"处理失败：{exc}")
            else:
                st.success("品牌范式提炼与候选图诊断完成")

                st.subheader("品牌样本评分总览")
                sample_table = [
                    {
                        "image_name": row.get("image_name", ""),
                        "balance_10": round(float(row.get("scores", {}).get("balance_10", 0.0)), 2),
                        "accessibility_10": round(float(row.get("scores", {}).get("accessibility_10", 0.0)), 2),
                        "final_10": round(float(row.get("scores", {}).get("final_10", 0.0)), 2),
                        "interactive_total": int(row.get("metrics", {}).get("interactive_total", 0)),
                    }
                    for row in sample_cases
                ]
                st.dataframe(sample_table, hide_index=True, use_container_width=True)

                summary_mode = str(brand_meta.get("mode", "none")) if isinstance(brand_meta, dict) else "none"
                summary_reason = str(brand_meta.get("reason", "")).strip() if isinstance(brand_meta, dict) else ""
                st.subheader("品牌设计范式总结")
                st.caption(
                    f"范式模式：{summary_mode}" + (f" | 原因：{summary_reason}" if summary_reason else "")
                )

                if brand_summary.get("brand_positioning"):
                    st.info(str(brand_summary.get("brand_positioning", "")))
                if brand_summary.get("pattern_summary"):
                    st.write(str(brand_summary.get("pattern_summary", "")))

                baseline = brand_summary.get("score_baseline", {}) if isinstance(brand_summary, dict) else {}
                if not isinstance(baseline, dict):
                    baseline = {}
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("基线平衡(0-10)", round(float(baseline.get("balance_mean", 0.0)), 2))
                b2.metric("基线可达(0-10)", round(float(baseline.get("accessibility_mean", 0.0)), 2))
                b3.metric("基线总分(0-10)", round(float(baseline.get("final_mean", 0.0)), 2))
                b4.metric("一致性", str(baseline.get("consistency", "medium")))

                patterns = brand_summary.get("design_patterns", []) if isinstance(brand_summary, dict) else []
                if isinstance(patterns, list) and patterns:
                    st.markdown("**提炼出的品牌范式**")
                    for idx, item in enumerate(patterns, start=1):
                        if not isinstance(item, dict):
                            continue
                        title = str(item.get("name", f"范式{idx}")).strip() or f"范式{idx}"
                        with st.expander(f"{idx}. {title}", expanded=(idx <= 2)):
                            if item.get("rule"):
                                st.write(f"规则：{item['rule']}")
                            evidence = item.get("evidence", []) if isinstance(item.get("evidence", []), list) else []
                            if evidence:
                                st.write("证据：")
                                for e_idx, evidence_item in enumerate(evidence[:6], start=1):
                                    st.write(f"{e_idx}. {evidence_item}")
                            if item.get("quant_signal"):
                                st.write(f"量化信号：{item['quant_signal']}")

                cols = st.columns(3)
                for title, key, col in [
                    ("Do", "dos", cols[0]),
                    ("Don't", "donts", cols[1]),
                    ("Checklist", "checklist", cols[2]),
                ]:
                    with col:
                        st.markdown(f"**{title}**")
                        items = brand_summary.get(key, []) if isinstance(brand_summary, dict) else []
                        if isinstance(items, list) and items:
                            for idx, item in enumerate(items[:8], start=1):
                                st.write(f"{idx}. {item}")
                        else:
                            st.write("暂无")

                st.subheader("候选图评分")
                candidate_score = candidate_result.metadata.get("score_10", {}) if isinstance(candidate_result.metadata, dict) else {}
                if not isinstance(candidate_score, dict):
                    candidate_score = {}
                c1, c2, c3 = st.columns(3)
                c1.metric("候选平衡(0-10)", candidate_score.get("physical_balance", 0.0))
                c2.metric("候选可达(0-10)", candidate_score.get("accessibility", 0.0))
                c3.metric("候选总分(0-10)", candidate_score.get("final", 0.0))

                diagnosis_mode = str(diagnosis_meta.get("mode", "none")) if isinstance(diagnosis_meta, dict) else "none"
                diagnosis_reason = str(diagnosis_meta.get("reason", "")).strip() if isinstance(diagnosis_meta, dict) else ""
                st.subheader("候选图相对品牌范式的差距诊断")
                st.caption(
                    f"诊断模式：{diagnosis_mode}" + (f" | 原因：{diagnosis_reason}" if diagnosis_reason else "")
                )

                alignment_map = {"high": "高匹配", "medium": "中匹配", "low": "低匹配"}
                alignment = str(diagnosis.get("overall_alignment", "medium")).lower().strip()
                st.write(f"匹配度：{alignment_map.get(alignment, '中匹配')}")
                if diagnosis.get("summary"):
                    st.info(str(diagnosis.get("summary", "")))

                score_alignment = diagnosis.get("score_alignment", {}) if isinstance(diagnosis.get("score_alignment", {}), dict) else {}
                g1, g2, g3 = st.columns(3)
                g1.metric("平衡差值(候选-基线)", round(float(score_alignment.get("balance_gap", 0.0)), 2))
                g2.metric("可达差值(候选-基线)", round(float(score_alignment.get("accessibility_gap", 0.0)), 2))
                g3.metric("总分差值(候选-基线)", round(float(score_alignment.get("final_gap", 0.0)), 2))
                if score_alignment.get("comment"):
                    st.write(f"分数解读：{score_alignment['comment']}")

                violations = diagnosis.get("violations", []) if isinstance(diagnosis.get("violations", []), list) else []
                if violations:
                    st.markdown("**主要不足（违反品牌范式）**")
                    sev_map = {"high": "高", "medium": "中", "low": "低"}
                    for idx, item in enumerate(violations[:10], start=1):
                        if not isinstance(item, dict):
                            continue
                        sev = sev_map.get(str(item.get("severity", "medium")).lower().strip(), "中")
                        text = f"{idx}. [{sev}] {item.get('issue', '')}"
                        if item.get("pattern"):
                            text += f"（对应范式：{item['pattern']}）"
                        st.write(text)
                        if item.get("evidence"):
                            st.caption(f"证据：{item['evidence']}")

                improvements = diagnosis.get("improvements", []) if isinstance(diagnosis.get("improvements", []), list) else []
                if improvements:
                    st.markdown("**改进建议（按优先级）**")
                    for idx, item in enumerate(improvements[:10], start=1):
                        if not isinstance(item, dict):
                            continue
                        title = str(item.get("title", f"优化项{idx}")).strip() or f"优化项{idx}"
                        priority = str(item.get("priority", "P1")).upper().strip() or "P1"
                        with st.expander(f"{idx}. [{priority}] {title}", expanded=(idx <= 2)):
                            if item.get("action"):
                                st.write(f"动作：{item['action']}")
                            if item.get("expected_gain"):
                                st.write(f"收益：{item['expected_gain']}")
                            if item.get("related_pattern"):
                                st.write(f"关联范式：{item['related_pattern']}")

                quick_wins = diagnosis.get("quick_wins", []) if isinstance(diagnosis.get("quick_wins", []), list) else []
                if quick_wins:
                    st.markdown("**Quick Wins**")
                    for idx, item in enumerate(quick_wins[:8], start=1):
                        st.write(f"{idx}. {item}")

                st.subheader("候选图可视化")
                candidate_artifacts = candidate_output.get("artifact_paths", {}) if isinstance(candidate_output, dict) else {}
                if isinstance(candidate_artifacts, dict) and candidate_artifacts:
                    vc1, vc2 = st.columns(2)
                    with vc1:
                        if candidate_artifacts.get("elements_annotated"):
                            st.image(candidate_artifacts["elements_annotated"], caption="组件识别与语义权重", use_container_width=True)
                        if candidate_artifacts.get("visual_balance_overlay"):
                            st.image(candidate_artifacts["visual_balance_overlay"], caption="视觉重心与几何中心", use_container_width=True)
                    with vc2:
                        if candidate_artifacts.get("reachability_heatmap"):
                            st.image(candidate_artifacts["reachability_heatmap"], caption="可达性热力图", use_container_width=True)
                        if candidate_artifacts.get("score_radar"):
                            st.image(candidate_artifacts["score_radar"], caption="维度雷达图", use_container_width=True)
