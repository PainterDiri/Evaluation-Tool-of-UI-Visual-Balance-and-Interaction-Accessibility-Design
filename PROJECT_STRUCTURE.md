# Project Structure

```text
Evaluation Tool/
├─ app.py                        # Streamlit 可视化入口
├─ main.py                       # CLI 入口
├─ requirements.txt              # 依赖
├─ .env.example                  # 环境变量模板
├─ .gitignore                    # Git 忽略规则
├─ README.md                     # 使用说明
├─ PROJECT_STRUCTURE.md          # 目录说明
├─ data/
│  └─ guidelines.json            # 评估规则示例
├─ input/
│  └─ .gitkeep                   # 示例输入目录（默认忽略实际截图）
├─ outputs/
│  └─ .gitkeep                   # 输出目录（报告与图像）
└─ src/
   ├─ __init__.py
   ├─ config.py                  # 运行配置与评分映射
   ├─ ui_types.py                # 数据结构定义
   ├─ image_analyzer.py          # OpenCV 组件提取与特征计算
   ├─ metrics.py                 # 平衡度/Fitts/可达性公式
   ├─ batch_analysis.py          # 批量评估与象限图绘制
   ├─ semantic_weighting.py      # LLM/启发式语义赋权
   ├─ guideline_manager.py       # 规则库加载
   ├─ visualization.py           # 标注图/热力图/雷达图绘制
   ├─ reporting.py               # JSON 报告导出
   └─ pipeline.py                # 统一评估流水线
```

## Execution Flow

1. 读取截图并提取候选组件
2. 计算面积、局部对比度、视觉权重
3. 语义赋权（LLM 或启发式）
4. 计算视觉平衡分与可达性分
5. 汇总综合得分与建议
6. 输出 JSON 与可视化文件
