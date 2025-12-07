import os
import base64
from pathlib import Path

import cv2
from dotenv import load_dotenv
from moviepy import VideoFileClip
from openai import OpenAI
import easyocr

# ====== 0) 初期化 ======
load_dotenv()
client = OpenAI()  # OPENAI_API_KEY は .env から読む

# グローバルに1回だけ初期化（毎回作ると遅いので）
OCR_READER = easyocr.Reader(['ja', 'en'], gpu=False)


# ====== テロップOCR ======
def extract_teletext_from_frame(frame_path: str) -> str:
    """
    1枚のフレームからテロップ文字を推定して返す。
    今は画面全体に対してOCRしている簡易版。
    """
    results = OCR_READER.readtext(frame_path, detail=0)  # detail=0 で文字列だけ返す
    # results は ["玉ねぎを5mm幅に切る", "カレーの具材"] みたいなリスト想定
    text = "\n".join(results)
    return text


# ====== 1) 動画 → フレーム抽出 ======
def extract_frames(video_path: str, output_dir: str, fps: int = 1) -> list[str]:
    """
    動画から 1秒あたり fps 枚のフレームを jpg で保存し、そのパス一覧を返す
    """
    # 既存のフレームを一旦消してクリーンに
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            fp = os.path.join(output_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)
    else:
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


# ====== 3) （今は未使用）音声 → テキスト（ASR） ======
def transcribe_audio(audio_path: str) -> str:
    """
    GPT-4o mini Transcribe で音声を文字起こし
    今はレート制限の都合で main からは呼んでいない。
    """
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
            # language="ja",
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
    VLM に
    - カット中のフレーム画像数枚
    - 音声の文字起こし（今は空）
    - 補足テキスト（例: テロップ内容）
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
        "画面内のテロップや文字があれば、その内容も読み取ってください。",
        "テロップに材料名や切り方、厚さの指定があれば、その情報を最優先で使ってください。",
        "動画やテロップの情報から、「何の食材を」「どのような切り方で」「どのくらいの厚さで」切るべきかを判断してください。",
        "初心者向けに、具体的な手順と安全面の注意を教えてください。",
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

                            "テロップの情報（材料名、分量、切り方、厚さの指定など）が与えられた場合は、"
                            "画像からの推測よりもテロップの情報を必ず優先してください。"
                            "特に、材料名（例: 玉ねぎ 1/2個）や『食べやすい大きさに切る』『5mm幅に切る』といった表現は、"
                            "できるだけテロップの文言をそのまま使ってください。\n"
                            "テロップに厚さや具体的な数値が書かれていない場合、"
                            "新たな数値（例: 1〜2cm や 5mm など）を勝手に作らないでください。"
                            "3番の『目標の厚さ』がテロップや画像から明確に判断できない場合は、"
                            "必ず「不明」とだけ書き、かっこ書きや追加説明は書いてはいけません。\n\n"

                            "動画中に複数の食材（例: 玉ねぎ、にんじん、じゃがいも）が順番に登場する場合、"
                            "それぞれの食材について同じ形式で指示を書いてください。"
                            "食材ごとに次の5項目セットを、【食材1】【食材2】…のように繰り返して出力してください。\n\n"

                            "【出力フォーマット】\n"
                            "各食材について、必ず次の5項目をこの順番で日本語で出力してください。\n"
                            "1. 今切るべき食材: （例: 玉ねぎ 1/2個。テロップに分量があれば含める。分からない場合は「不明」と書く）\n"
                            "2. 切り方の種類: （例: 薄切り / いちょう切り / 乱切り / 食べやすい大きさに切る など。テロップの表現を優先する）\n"
                            "3. 目標の厚さ: （例: 約5mm など。テロップに厚さの指定がある場合のみ書く。指定がない場合は「不明」とだけ書く）\n"
                            "4. 手順: 初心者向けに、どの向きに置いて、どの方向に包丁を動かし、どのくらいの幅で切るかを、2〜4文で具体的に説明する。\n"
                            "   ただし、3番が「不明」の場合は、4番の中でも mm や cm などの具体的な数値を使ってはいけません。"
                            "   その場合は『食べやすい大きさ』『一口大』『大きめにそろえる』など、テロップと矛盾しない曖昧な表現のみを用いてください。\n"
                            "5. 安全面の注意: 指の置き方・刃の向きなど、安全のために特に気をつけるポイントを1〜2文で書く。\n\n"

                            "これは特定の料理名（カレーなど）に限られた動画ではなく、"
                            "さまざまな料理の『食材を切る工程』を含むクリップとして扱ってください。"
                            "料理名を推測する必要はなく、純粋に画面の食材・テロップ・切り方の情報に基づいて指示を生成してください。\n\n"

                            "もし動画の冒頭や一部に『完成した料理（盛り付け後）』が映っている場合は、"
                            "その見た目から食材のおおよその大きさ・形状・切り方の傾向を推測してかまいません。"
                            "ただし、その推測は『この料理では具材を大きめに切る傾向がある』『細めの棒状に切られている』など、"
                            "仕上がりの特徴を自然言語で補足する形にとどめ、mm や cm などの具体的な数値を勝手に作らないでください。"
                            "厚さが不明の場合でも、完成料理を参考に『ゴロっとした一口大が好ましい』『薄めで火の通りやすい形が多い』など、"
                            "料理の仕上がりイメージに合わせた表現を4番の手順の中に含めてよいものとします。\n"
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

    parts = []
    for item in response.output[0].content:
        if item.type == "output_text":
            parts.append(item.text)

    return "\n".join(parts)


# ====== 5) 1クリップ（1食材分）を解析する関数 ======
def analyze_cut_clip(video_path: str, extra_hint: str | None = None) -> str:
    """
    1本の「カットシーン動画」に対して:
    - フレーム抽出
    - テロップOCR
    - （音声抽出・文字起こしは今はスキップ）
    - VLM で「食材名・切り方・厚さ・手順・安全」を生成
    をまとめて行う。
    """
    frames_dir = "frames_tmp"
    audio_path = "audio.wav"

    print(f"[INFO] 動画: {video_path}")

    # 1) フレーム抽出（情報量を増やしたいので fps=3 に）
    print("[INFO] フレーム抽出中 ...")
    frames = extract_frames(video_path, frames_dir, fps=1)

    if not frames:
        print("[ERROR] フレームが抽出できませんでした")
        return "解析に失敗しました（フレームなし）"

    # 最初の10枚くらいを使う
    rep_frames = frames[:21]
    print(f"[INFO] 使用フレーム枚数: {len(rep_frames)}")

    # 2) テロップOCR（代表フレームの中から数枚ピックして読む）
    teletexts = []
    candidate_idxs = [
        len(rep_frames) // 4,
        len(rep_frames) // 2,
        (3 * len(rep_frames)) // 4,
    ]
    candidate_idxs = sorted(set(i for i in candidate_idxs if 0 <= i < len(rep_frames)))

    print("[INFO] テロップOCR中 ...")
    for fp in rep_frames:
        t = extract_teletext_from_frame(fp)
        if t.strip():
            teletexts.append(t)

    # 重複を削る（同じテロップを何度も読むのを避ける）
    teletexts = list(dict.fromkeys(teletexts))

    teletext_merged = "\n".join(teletexts)
    print("----- OCR で取得したテロップ候補 -----")
    print(teletext_merged if teletext_merged else "(何も読み取れませんでした)")

    # 3) 音声抽出（今はASRを使わない前提でもOK）
    print("[INFO] 音声抽出中 ...")
    extract_audio(video_path, audio_path)

    print("[INFO] 音声の文字起こし中 現在スキップ ...")
    transcript = ""  # transcribe_audio(audio_path) に差し替えれば音声も使える
    print("----- 文字起こし結果 現在スキップ-----")

    # 4) extra_hint にテロップ情報などを統合
    hint_parts = []
    if extra_hint:
        hint_parts.append(extra_hint)
    if teletext_merged:
        hint_parts.append(
            "画面のテロップには次のように書かれています。"
            "材料名や切り方の指定があれば、この情報を最優先に使ってください:\n"
            + teletext_merged
        )

    merged_hint = "\n\n".join(hint_parts) if hint_parts else None

    # 5) VLM に投げる
    print("[INFO] VLM へ問い合わせ中 ...")
    instruction = get_instruction_from_vlm(
        frame_paths=rep_frames,
        audio_transcript=transcript,
        extra_hint=merged_hint,
    )

    return instruction


# ====== 6) メイン処理：複数クリップ（複数食材）を順番に解析 ======
def main():
    movies_dir = Path("movies")

    # 単一ファイルだけ試したい場合はここを書き換えてもOK
    # video_paths = [movies_dir / "curry_cut_of_delishkitchen.mov"]

    # movies ディレクトリ内の .mov / .mp4 を全部対象にする
    video_paths = sorted(
        list(movies_dir.glob("*.mov")) + list(movies_dir.glob("*.mp4"))
    )

    if not video_paths:
        print("[ERROR] movies ディレクトリに動画がありません")
        return

    print(f"[INFO] 解析対象の動画本数: {len(video_paths)}")

    for idx, vp in enumerate(video_paths, start=1):
        print("\n" + "=" * 50)
        print(f"[INFO] 食材 {idx} 本目の動画: {vp.name}")

        extra_hint = (
            "このシーンは、家庭料理におけるこの動画は料理の下ごしらえにおける複数の食材のカットシーンを含んでいます。"
        )

        instruction = analyze_cut_clip(str(vp), extra_hint=extra_hint)

        print("\n===== この食材への指示 =====")
        print(instruction)
        print("=" * 50)


if __name__ == "__main__":
    main()
