"""Ручной запуск инференса"""
import argparse
from pathlib import Path

import torch

from inference.model_loader import load_multilabel_model
from inference.recommender import recommend_tracks_for_video
from inference.text_encoder import load_text_encoder
from inference.video_encoder import load_video_encoder


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", required=True)
    parser.add_argument("--video_title", required=True)
    parser.add_argument("--user_query", required=True)

    parser.add_argument(
        "--checkpoint_path",
        default="model_weights\\best_target_binary_multilabel_model_inference.pt",
    )

    parser.add_argument(
        "--project_root",
        default=r"C:\VS code projects\NLP lingua of internet\cv_project",
    )

    parser.add_argument(
        "--text_model_name",
        default="intfloat/multilingual-e5-small",
    )

    parser.add_argument(
        "--video_model_name",
        default="MCG-NJU/videomae-base-finetuned-kinetics",
    )

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_video_frames", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--hf_pooling", default="mean")

    parser.add_argument("--query_input_dim", type=int, default=384)
    parser.add_argument("--title_input_dim", type=int, default=384)
    parser.add_argument("--video_input_dim", type=int, default=768)

    parser.add_argument("--text_dim", type=int, default=256)
    parser.add_argument("--video_dim", type=int, default=256)
    parser.add_argument("--duration_dim", type=int, default=32)
    parser.add_argument("--fusion_hidden_dim", type=int, default=512)
    parser.add_argument("--fusion_output_dim", type=int, default=256)

    parser.add_argument(
        "--output_csv",
        default="training_artifacts/manual_inference_multilabel_predictions.csv",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, track2idx, idx2track = load_multilabel_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        query_input_dim=args.query_input_dim,
        title_input_dim=args.title_input_dim,
        video_input_dim=args.video_input_dim,
        text_dim=args.text_dim,
        video_dim=args.video_dim,
        duration_dim=args.duration_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        fusion_output_dim=args.fusion_output_dim,
    )

    tokenizer, text_model = load_text_encoder(
        model_name=args.text_model_name,
        device=device,
    )

    video_processor, video_model = load_video_encoder(
        model_name=args.video_model_name,
        device=device,
        trust_remote_code=True,
    )

    result_df = recommend_tracks_for_video(
        video_path=args.video_path,
        video_title=args.video_title,
        user_query=args.user_query,
        model=model,
        tokenizer=tokenizer,
        text_model=text_model,
        video_processor=video_processor,
        video_model=video_model,
        idx2track=idx2track,
        device=device,
        project_root=args.project_root,
        top_k=args.top_k,
        num_video_frames=args.num_video_frames,
        image_size=args.image_size,
        hf_pooling=args.hf_pooling,
    )

    print(result_df)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()