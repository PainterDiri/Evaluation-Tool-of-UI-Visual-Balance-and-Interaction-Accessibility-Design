from __future__ import annotations

import argparse
from pathlib import Path

from src.batch_analysis import run_batch_analysis
from src.config import RuntimeConfig
from src.pipeline import EvaluationPipeline
from src.paths import DEFAULT_GUIDELINE_FILE, OUTPUT_DIR, list_input_images, resolve_default_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate mobile UI visual balance and interaction accessibility"
    )
    parser.add_argument("--image", default=None, help="Path to UI screenshot. Defaults to the first image in input/")
    parser.add_argument("--guidelines", default=str(DEFAULT_GUIDELINE_FILE), help="Path to guideline JSON")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output folder for report and visuals")
    parser.add_argument(
        "--origin",
        default="bottom-center",
        choices=["bottom-center", "bottom-right"],
        help="Thumb origin mode for Fitts calculation",
    )
    parser.add_argument("--use-llm", action="store_true", help="Enable multimodal LLM semantic weighting")
    parser.add_argument("--max-elements", type=int, default=180, help="Upper bound of detected components")
    parser.add_argument("--min-area-ratio", type=float, default=0.0015, help="Min bbox area ratio")
    parser.add_argument(
        "--top-crop-ratio",
        type=float,
        default=0.04,
        help="Crop ratio for top status bar suppression (0 disables)",
    )
    parser.add_argument(
        "--accessibility-max-id",
        type=float,
        default=6.2,
        help="Upper ID bound used to map weighted Fitts cost to accessibility score",
    )
    parser.add_argument("--batch", action="store_true", help="Evaluate all images in input/ and export quadrant plots")
    parser.add_argument(
        "--batch-save-individual",
        action="store_true",
        help="When --batch is enabled, also save per-image report and visualization artifacts",
    )
    parser.add_argument(
        "--batch-balance-threshold",
        type=float,
        default=0.75,
        help="Vertical split line for quadrant classification (0-1)",
    )
    parser.add_argument(
        "--batch-access-threshold",
        type=float,
        default=0.65,
        help="Horizontal split line for quadrant classification (0-1)",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save report and visualization files")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = RuntimeConfig(
        origin_mode=args.origin,
        use_llm=args.use_llm,
        output_dir=Path(args.output_dir),
        max_elements=args.max_elements,
        min_area_ratio=args.min_area_ratio,
        top_crop_ratio=args.top_crop_ratio,
        accessibility_max_reasonable_id=args.accessibility_max_id,
    )

    pipeline = EvaluationPipeline(config=config, guideline_file=args.guidelines)

    if args.batch:
        image_paths = list_input_images()
        if not image_paths:
            raise FileNotFoundError("No images found in input/ for batch analysis")

        batch_output = run_batch_analysis(
            pipeline=pipeline,
            image_paths=image_paths,
            output_dir=config.output_dir,
            save_individual_artifacts=args.batch_save_individual,
            x_threshold=args.batch_balance_threshold,
            y_threshold=args.batch_access_threshold,
        )

        print("=" * 72)
        print("Batch Evaluation Summary")
        print("=" * 72)
        print(f"Images analyzed: {batch_output['count']}")
        print("Quadrant counts:")
        for quadrant, count in batch_output["quadrant_counts"].items():
            print(f"  - {quadrant}: {count}")
        print(f"Summary JSON: {batch_output['summary_path']}")
        print(f"Quadrant plot: {batch_output['quadrant_plot']}")
        print(f"Quadrant plot with thumbnails: {batch_output['quadrant_plot_with_thumbnails']}")
        return

    image_path = Path(args.image) if args.image else resolve_default_image()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output = pipeline.run(image_path=image_path, save_artifacts=not args.no_save)
    result = output["result"]

    print("=" * 72)
    print("Evaluation Summary")
    print("=" * 72)
    print(f"Image: {result.image_path}")
    print(f"Resolution: {result.width}x{result.height}")
    print(f"Elements: total={result.elements_total}, interactive={result.interactive_total}")
    print(
        "Scores (0-10): "
        f"Balance={result.metadata['score_10']['physical_balance']}, "
        f"Accessibility={result.metadata['score_10']['accessibility']}, "
        f"Final={result.metadata['score_10']['final']}"
    )
    print(f"Weighted interaction cost: {result.score_bundle.weighted_interaction_cost:.3f}")
    print("Recommendations:")
    for i, rec in enumerate(result.recommendations, start=1):
        print(f"  {i}. {rec}")

    detail_mode = result.metadata.get("detailed_analysis_mode", "none")
    detail_reason = result.metadata.get("detailed_analysis_reason", "")
    detail = result.metadata.get("detailed_analysis", {})
    if isinstance(detail, dict) and detail:
        print("\nDetailed Diagnosis:")
        print(f"  mode: {detail_mode}")
        summary = str(detail.get("overall_summary", "")).strip()
        if summary:
            print(f"  summary: {summary}")

        vb = detail.get("visual_balance", {})
        if isinstance(vb, dict):
            print(f"  visual_balance.level: {vb.get('level', 'medium')}")
            if vb.get("why"):
                print(f"  visual_balance.why: {vb['why']}")

        ia = detail.get("interaction_accessibility", {})
        if isinstance(ia, dict):
            print(f"  interaction_accessibility.level: {ia.get('level', 'medium')}")
            if ia.get("why"):
                print(f"  interaction_accessibility.why: {ia['why']}")

        plan = detail.get("optimization_plan", [])
        if isinstance(plan, list) and plan:
            print("  optimization_plan:")
            for idx, item in enumerate(plan, start=1):
                if not isinstance(item, dict):
                    continue
                print(f"    {idx}. [{item.get('priority', 'P1')}] {item.get('title', '')}")
                if item.get("problem"):
                    print(f"       problem: {item['problem']}")
                if item.get("actions"):
                    print(f"       actions: {' | '.join(item['actions'])}")
    elif detail_mode not in {"none", "skipped"} and detail_reason:
        print(f"\nDetailed diagnosis fallback reason: {detail_reason}")

    if output["report_path"]:
        print(f"\nJSON report: {output['report_path']}")
    if output["artifact_paths"]:
        print("Visual artifacts:")
        for k, v in output["artifact_paths"].items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
