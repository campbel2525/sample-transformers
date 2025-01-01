from typing import Any, Dict, List

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


def _get_model_class(model_name: str, task_name: str):
    """
    モデル名 (e.g. "sonoisa/t5-base-japanese", "matsuo-lab/weblab-10b", etc.)
    と タスク名 (e.g. "text-generation", "summarization", "sentiment-analysis")
    を指定すると、

    - model_type (t5, gpt2, distilbert など) を自動判別
    - タスク名から想定されるデフォルトの ModelForXXX クラスを決定
    - model_type とタスク名の組み合わせに基づき最終的に使うクラスを返す

    だけを行う最小のサンプルです。
    """
    # config だけをロードして、model_type を取り出す
    config = AutoConfig.from_pretrained(model_name)
    model_type = config.model_type.lower()

    # モデルタイプによるモデルクラス
    class_map_for_model_type = {
        # エンコーダ・デコーダ系
        "t5": AutoModelForSeq2SeqLM,
        "bart": AutoModelForSeq2SeqLM,
        "mbart": AutoModelForSeq2SeqLM,
        "pegasus": AutoModelForSeq2SeqLM,
        "bigbird_pegasus": AutoModelForSeq2SeqLM,
        # デコーダ単体系
        "gpt2": AutoModelForCausalLM,
        "gpt_neo": AutoModelForCausalLM,
        "bloom": AutoModelForCausalLM,
        "llama": AutoModelForCausalLM,
        "opt": AutoModelForCausalLM,
        "mpt": AutoModelForCausalLM,
        # BERT系 (SequenceClassification が多い例)
        "bert": AutoModelForSequenceClassification,
        "distilbert": AutoModelForSequenceClassification,
        "roberta": AutoModelForSequenceClassification,
        # 必要に応じて他の model_type も追記
        "gpt_neox": AutoModelForCausalLM,  # GPT-NeoX は CausalLM
    }

    if model_type in class_map_for_model_type:
        return class_map_for_model_type[model_type]

    # デフォルトクラス
    default_class_map = {
        "text-generation": AutoModelForCausalLM,
        "summarization": AutoModelForSeq2SeqLM,
        "sentiment-analysis": AutoModelForSequenceClassification,
        # 必要に応じて増やす (question-answering, token-classification, etc.)
    }
    if task_name in default_class_map:
        return default_class_map[task_name]

    # どのクラスも該当しない場合はエラー
    raise ValueError(f"Unsupported model type: {model_type}")


def summary_sentences(
    sentences: List[str],
    model_name: str,
) -> List[str]:
    """
    要約を行う
    """

    task_name = "summarization"

    # モデルのロード
    ModelClass = _get_model_class(model_name, task_name)
    model = ModelClass.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
        offload_folder="/tmp",
    )

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # パイプラインの作成
    summarizer = pipeline(task_name, model=model, tokenizer=tokenizer)

    # サマリーの生成
    summaries = summarizer(sentences, min_length=1, max_length=1000)

    return [summary["summary_text"] for summary in summaries]


def embedding_sentences(
    sentences: List[str],
    model_name: str,
) -> List[List[float]]:
    """
    ベクトル化を行う
    """

    # 1. cl-tohoku/bert-base-japanese-whole-word-masking
    # このモデルは全単語マスキングを行ったBERTモデルで、単語レベルでの意味を捉えることができます。
    # したがって、単語の意味が重要なタスク（例えば、文章の意味理解や質問応答など）に適しています。
    #
    # 2. cl-tohoku/bert-base-japanese-char
    # このモデルは文字レベルでBERTモデルを訓練したもので、
    # 細かい文字レベルの情報を必要とするタスク（例えば、品詞タグ付けや固有表現抽出など）に適しています。

    model = SentenceTransformer(model_name)
    sentence_embeddings = model.encode(sentences)

    return [sentence_embedding.tolist() for sentence_embedding in sentence_embeddings]


def sentiment_sentences(
    sentences: List[str],
    model_name: str,
) -> List[Dict[str, Any]]:
    """
    感情分析を行う
    """

    task_name = "sentiment-analysis"

    # モデルのロード
    ModelClass = _get_model_class(model_name, task_name)
    model = ModelClass.from_pretrained(
        model_name,
    )

    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # パイプラインの作成
    classifier = pipeline(task_name, model=model, tokenizer=tokenizer)

    # 感情分析の実行
    classifier_results = classifier(sentences)
    return classifier_results
