import os
import base64
from pathlib import Path

import cv2
from dotenv import load_dotenv
from moviepy import VideoFileClip
from openai import OpenAI

# ====== 0) 初期化 ======
load_dotenv()
client = OpenAI()  # OPENAI_API_KEY は .env から読む

# ====== 1) 動画 → フレーム抽出 ======
def extract_frames(video_path: str, output_dir: str, fps: int = 1) -> list[str]:
    """
    動画から 1秒あたり fps 枚のフレームを jpg で保存し、そのパス一覧を返す
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = fps  # 何かおかしくても一応回す

    frame_interval = int(original_fps / fps) if original_fps > fps else 1

    frames = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_idx:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return frames


def select_representative_frames(frame_paths: list[str], num_samples: int = 3) -> list[str]:
    """
    先頭・中間・末尾 など、代表フレームを num_samples 枚選ぶ
    """
    if len(frame_paths) <= num_samples:
        return frame_paths

    idxs = [
        0,
        len(frame_paths) // 2,
        len(frame_paths) - 1,
    ][:num_samples]

    return [frame_paths[i] for i in idxs]


# ====== 2) 動画 → 音声抽出 ======
def extract_audio(video_path: str, audio_path: str) -> str:
    """
    moviepy を使って動画から音声のみを抽出（wav）
    """
    clip = VideoFileClip(video_path)
    audio_clip = clip.audio
    audio_clip.write_audiofile(audio_path, codec="pcm_s16le")
    audio_clip.close()
    clip.close()
    return audio_path


# ====== 3) 音声 → テキスト（ASR） ======
def transcribe_audio(audio_path: str) -> str:
    """
    GPT-4o mini Transcribe で音声を文字起こし
    """
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # 音声→テキスト用モデル :contentReference[oaicite:2]{index=2}
            file=f,
            # language="ja",  # 日本語のみなら指定してもよい
        )
    return transcription.text


# ====== 4) 画像 + テキスト → 指示生成 (VLM) ======
def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # 画像が jpg なら image/jpeg、png なら image/png にしてOK
    return f"data:image/jpeg;base64,{b64}"


def get_instruction_from_vlm(
    frame_paths: list[str],
    audio_transcript: str | None = None,
    extra_hint: str | None = None,
) -> str:
    """
    GPT-4o-mini (Text+Vision) に、
    - カット中のフレーム画像数枚
    - 音声の文字起こし
    - 補足テキスト（例: 「玉ねぎを5mm幅に切る場面です」）
    を渡して、「今何をするべきか」指示をもらう
    """
    user_content = []

    # 画像をメッセージに追加
    for p in frame_paths:
        data_url = encode_image_to_data_url(p)
        user_content.append(
            {
                "type": "input_image",
                "image_url": data_url,
            }
        )

    # テキスト入力部分
    text_parts = [
        "これらの画像は、料理動画の中の『具材カット中』のシーンを抜き出したものです。",
        "現在の手元の状態を簡単に説明したうえで、",
        "初心者向けに『今やるべきこと』を日本語で具体的に指示してください。",
        "指示は2〜4文程度で、行動ベースで書いてください。",
        "また、安全面（指の位置など）についても必ず1文コメントしてください。",
    ]

    if extra_hint:
        text_parts.append(f"補足情報: {extra_hint}")

    if audio_transcript:
        text_parts.append("動画内の音声の文字起こしは次の通りです。内容も参考にしてください。")
        text_parts.append(f"音声の文字起こし: {audio_transcript}")

    user_content.append(
        {
            "type": "input_text",
            "text": "\n".join(text_parts),
        }
    )

    # GPT-4o-mini Vision に投げる :contentReference[oaicite:3]{index=3}
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "あなたはプロの料理講師です。"
                            "ユーザーが行っている『具材のカット』の動画フレームと、"
                            "画面内のテロップなどの文字情報から、何をどのように切るべきかを判断し、"
                            "初心者向けに非常に具体的な指示を行ってください。\n\n"
                            "画像に写っている食材を必ず推定してください。色・形・皮の質感から、可能な限り具体的な野菜名を推定してください。分からないときでも推測を試みてください。"
                            "出力フォーマットは必ず次の5項目をこの順番で日本語で出力してください:\n"
                            "1. 今切るべき食材: （例: 玉ねぎ / にんじん など。分からない場合は「不明」と書く）\n"
                            "2. 切り方の種類: （例: 薄切り / いちょう切り / 乱切り など。分からない場合は「不明」と書く）\n"
                            "3. 目標の厚さ: （例: 約5mm など。画面上に指定がない場合は「不明」と書く）\n"
                            "4. 手順: 初心者向けに、どの向きに置いて、どの方向に包丁を動かし、どのくらいの幅で切るかを、2〜4文で具体的に説明する。\n"
                            "5. 安全面の注意: 指の置き方・刃の向きなど、安全のために特に気をつけるポイントを1〜2文で書く。\n\n"
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )

    # responses API の構造に沿ってテキストを取り出す
    # （output[0].content[0].text 形式 :contentReference[oaicite:4]{index=4}）
    parts = []
    for item in response.output[0].content:
        if item.type == "output_text":
            parts.append(item.text)

    return "\n".join(parts)


# ====== 5) メイン処理 ======
def main():
    # ---- ここを書き換えて、自分の動画ファイルを指定 ----
    video_path = "movies/curry_cut_of_delishkitchen.mov"
    video_path = str(Path(video_path).resolve())

    frames_dir = "frames"
    audio_path = "audio.wav"

    print(f"[INFO] 動画: {video_path}")

    # 1) フレーム抽出
    print("[INFO] フレーム抽出中 ...")
    frames = extract_frames(video_path, frames_dir, fps=1)

    if not frames:
        print("[ERROR] フレームが抽出できませんでした")
        return

    rep_frames = frames[3:21]
    print(f"[INFO] 代表フレーム枚数: {len(rep_frames)}")

    # 2) 音声抽出
    print("[INFO] 音声抽出中 ...")
    extract_audio(video_path, audio_path)

    # 3) 音声文字起こし
    print("[INFO] 音声の文字起こし中 現在スキップ ...")
    # transcript = transcribe_audio(audio_path)
    print("----- 文字起こし結果 現在スキップ-----")
    # print(transcript)

    # 4) VLMに投げて指示をもらう
    # extra_hint に「玉ねぎを5mm幅に切る場面です」などを入れてもOK
    extra_hint = "このシーンは、カレー用の野菜を切る工程です。一般的な家庭用カレーの下ごしらえとして想定してください。"
    print("[INFO] VLM へ問い合わせ中 ...")
    instruction = get_instruction_from_vlm(
        frame_paths=rep_frames,
        # audio_transcript=transcript,
        extra_hint=extra_hint,
    )

    print("\n===== VLM からの指示 =====")
    print(instruction)


if __name__ == "__main__":
    main()
