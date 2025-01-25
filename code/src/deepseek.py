"""
- huggingface hubのモデルダウンロードして使用するサンプル

"""

from services.transformer_services import download_model, run_model


if __name__ == "__main__":
    # # モデルをlocalにダウンロード
    # download_model(
    #     repo_id="deepseek-ai/DeepSeek-R1",
    #     local_dir="model_repo/models/DeepSeek-R1",
    # )

    # モデルの実行
    prompt = "こんにちは"
    result = run_model(
        model_name="model_repo/models/DeepSeek-R1",
        prompt=prompt,
    )
    print(result)
