# Evaluation Tool: UI Visual Balance & Interaction Accessibility

一个面向移动端截图的量化评估工具，采用 Hybrid Pipeline：
- OpenCV 负责组件几何提取与物理计算
- 可选 MLLM 负责交互元素语义优先级赋权

输出包括：
- 视觉平衡度（Physical Visual Balance）
- 交互可达性（Fitts + Semantic Weight）
- 综合得分（0-10）
- JSON 报告与多张可视化图
- 启用 LLM 时的结构化详细诊断：
	- 视觉平衡为什么高/低
	- 交互可达性为什么高/低
	- 分优先级的详细优化方案（含步骤、预期收益、验证方式）

## 1. 核心能力

1. 自动提取 UI 候选组件（bbox、面积、局部对比度）
2. 根据视觉权重计算界面重心并输出平衡分
3. 基于拇指起点模型计算 Fitts 交互成本
4. 融合语义权重得到加权可达性成本
5. 生成改进建议、结构化报告和图像可视化
6. 同品牌多样本范式提炼（结合分数与截图内容）
7. 候选页面对照品牌范式的差距诊断与改进建议

## 2. 安装

```bash
pip install -r requirements.txt
```

如果需要 LLM 语义赋权，复制环境变量模板：

```bash
copy .env.example .env
```

然后填写：
- LLM_API_KEY
- LLM_BASE_URL
- LLM_MODEL

未配置时，系统自动使用启发式语义赋权（仍可完整运行）。

## 3. CLI 运行

```bash
python main.py --image path/to/ui.png --output-dir outputs
```

常用参数：
- --origin bottom-center|bottom-right
- --use-llm
- --min-area-ratio 0.0015
- --max-elements 180
- --top-crop-ratio 0.055
- --accessibility-max-id 6.2
- --batch
- --batch-balance-threshold 0.75
- --batch-access-threshold 0.65
- --no-save

示例：

```bash
python main.py --use-llm
```

如果不指定 `--image`，程序会自动读取 `input/` 目录下排序后的第一张图片作为默认输入。

批量评估（将 `input/` 所有图片绘制到象限图）：

```bash
python main.py --batch --output-dir outputs
```

可选：
- `--batch-save-individual`：批处理时也保存每张图的单图报告与可视化
- `--batch-balance-threshold`：象限图竖线阈值（默认 0.75）
- `--batch-access-threshold`：象限图横线阈值（默认 0.65）

## 4. Streamlit 可视化运行

```bash
streamlit run app.py
```

页面支持：
- 上传截图
- 设置拇指起点与检测参数
- 一键评估
- 一键批量评估 `input/` 并绘制象限图（编号映射 + 缩略图）
- 同品牌多图范式提炼 + 新图对照诊断（建议启用 LLM）
- 在线查看标注图、重心图、热力图、雷达图
- 下载 JSON 报告

品牌范式功能说明：
- 先选择同品牌样本图（建议 3-8 张，可从 `input/` 多选或直接上传）
- 工具会先逐张评估并汇总分数，然后结合截图内容提炼品牌设计范式
- 再选择一张候选页面，工具会先评分，再输出其相对品牌范式的不足与可执行优化建议
- 若未启用 LLM，将自动回退到统计规则模式（仍可输出基线与差距）

## 5. 结果说明

报告文件：
- outputs/evaluation_report.json

关键字段：
- scores.physical_balance：范围 [0,1]
- scores.accessibility：范围 [0,1]
- scores.weighted_interaction_cost：越低越好
- metadata.score_10：三项 0-10 分
- metadata.detailed_analysis_mode：详细诊断模式（llm-detailed/skipped/fallback）
- metadata.detailed_analysis：结构化详细诊断结果（启用 LLM 时）

结构化详细诊断（metadata.detailed_analysis）包含：
- overall_summary：总体解释
- visual_balance：视觉平衡等级、原因、证据、规则解释
- interaction_accessibility：可达性等级、原因、证据、规则解释
- optimization_plan：按 P0/P1/P2 排序的优化计划（目标组件、问题、设计规则、执行步骤、预期效果、验证方式）

可视化文件：
- outputs/elements_annotated.png
- outputs/visual_balance_overlay.png
- outputs/reachability_heatmap.png
- outputs/reachability_blend.png
- outputs/score_radar.png
- outputs/batch_quadrant_plot.png
- outputs/batch_quadrant_plot_thumbnails.png

批量报告文件：
- outputs/batch_evaluation_summary.json

## 6. 方法与公式

视觉平衡度：
- Wi = Areai * Contrasti
- Xcm = sum(Wi * xi) / sum(Wi)
- Ycm = sum(Wi * yi) / sum(Wi)
- BM = 1 - dist((Xcm, Ycm), (Xc, Yc)) / dist((0,0), (Xc, Yc))

交互难度：
- IDi = log2(Di / Si + 1)

语义加权成本：
- Cost_weighted = sum(alpha_i * IDi) / sum(alpha_i)

## 7. 可扩展方向

1. 对接 OCR，将文本内容纳入语义判断
2. 引入目标检测模型替代轮廓检测，提高组件识别精度
3. 增加批处理模式，直接评估 50 张样本并输出象限图
4. 新增专家标注对齐模块，自动计算 Pearson/Spearman 相关系数
