import logging
import os
from argparse import Namespace

import datasets
import datasets.utils.logging
import torch
import transformers
import transformers.utils.logging
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

# =====================================================================
# 定数宣言
# =====================================================================
# pcのスペックが足りなく実行できない場合は下記の値を小さくして対応
# - config_dict
#   - train_dataset_size
#   - train_batch_size
#   - valid_batch_size
#   - seq_length

# データセット名(書籍: transformersbook/codeparrot)
# 書籍で用意されているデータセット名を使用する
TRAIN_DATASET_NAME = "llm-book/aio-passages-bpr-bert-base-japanese-v3"

# 学習に使用するモデル名
MODEL_NAME = "gpt2"

# ベース語彙
BASE_VOCAB = bytes_to_unicode()

# プロジェクト名
PROJECT_NAME = "myproject2"

# トークナイザーの保存先(local)
TOKENIZER_SAVE_DIR = f"model_repo/tokenizers/{PROJECT_NAME}/"

# モデルの保存先(local)
MODEL_SAVE_DIR = f"model_repo/models/{PROJECT_NAME}/"

# モデルの実行する際に使用するトークナイザー
EXECUTE_TOKENIZER_DIR = TOKENIZER_SAVE_DIR

# モデルの実行する際に使用するモデル名
EXECUTE_MODEL_DIR = f"{MODEL_SAVE_DIR}model_checkpoint_step_160/"

# トークナイザー名
# hugging faceのトークナイザーを指定することも可能
# 今回はlocalに保存しているトークナイザーを指定
TOKENIZER_NAME = TOKENIZER_SAVE_DIR

# ログの保存先
LOG_DIR = "logs/"

# Accelerate などで使用されるパラメータ
config_dict = {
    # データセットのカラム(書籍: "content")
    "dataset_use_column": "text",
    # 語彙サイズ(書籍: 12500)
    "vocab_size_large": 12500,
    # 学習データの長さ(書籍: 100000 or 200000)
    "train_dataset_size": 100000,
    # 学習時のバッチサイズ(書籍:2)
    "train_batch_size": 2,
    # 検証時のバッチサイズ(書籍:2)
    "valid_batch_size": 2,
    # Weight Decay（ウェイト減衰、L2 正則化）の係数(書籍:0.1)
    "weight_decay": 0.1,
    # ストリーミングデータセットのシャッフル時に使用するバッファサイズ(書籍:1000)
    "shuffle_buffer": 1000,
    # 学習率(書籍: 2e-4)
    "learning_rate": 2e-4,
    # 学習率を徐々に変動させるスケジューラの種類(書籍: cosine)
    "lr_scheduler_type": "cosine",
    # ウォームアップステップ数(書籍: 750)
    "num_warmup_steps": 750,
    # 勾配の累積ステップ数(書籍: 16)
    "gradient_accumulation_steps": 16,
    # 学習の最大ステップ数(書籍: 50000)
    "max_train_steps": 10,
    # 検証の際の最大ステップ数(書籍: -1 -1: すべて)
    "max_eval_steps": 10,
    # 入力シーケンスの最大長(書籍: 1024)
    "seq_length": 1024,
    # 乱数シードを固定するためのシード値(書籍: 1)
    "seed": 1,
    # チェックポイントを保存するステップ間隔(書籍: 50000)
    "save_checkpoint_steps": 10,
}
args = Namespace(**config_dict)


# =====================================================================
# トークナイザーの学習
# =====================================================================
def train_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ストリーミングデータセットを用いて学習
    dataset = load_dataset(TRAIN_DATASET_NAME, split="train", streaming=True)
    iter_dataset = iter(dataset)

    def batch_iterator_larger(batch_size=10):
        for _ in tqdm(range(0, args.train_dataset_size, batch_size)):
            yield [
                next(iter_dataset)[args.dataset_use_column] for _ in range(batch_size)
            ]

    new_tokenizer = tokenizer.train_new_from_iterator(
        batch_iterator_larger(),
        vocab_size=args.vocab_size_large,
        initial_alphabet=BASE_VOCAB,
    )

    # トークナイザーをローカルに保存
    os.makedirs(TOKENIZER_SAVE_DIR, exist_ok=True)
    new_tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)

    return new_tokenizer


# =====================================================================
# モデルの学習
# =====================================================================
class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        while True:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    break
                try:
                    buffer.append(next(iterator)[args.dataset_use_column])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    # もう一度イテレータを作り直す
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(
                buffer,
                truncation=False,
                max_length=self.seq_length,
            )
            for tokenized_input in tokenized_inputs["input_ids"]:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]  # noqa
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)


def create_dataloaders(tokenizer):
    """
    目的: 学習用/検証用データを読み込み、トークナイズ & 定長に切り出し、PyTorch の DataLoader にして返す。
    流れ: load_dataset → シャッフル → ConstantLengthDataset → DataLoader 生成 → return。
    """

    train_data = load_dataset(
        TRAIN_DATASET_NAME + "-train", split="train", streaming=True
    )
    train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)

    valid_data = load_dataset(
        TRAIN_DATASET_NAME + "-valid", split="validation", streaming=True
    )

    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, seq_length=args.seq_length
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, seq_length=args.seq_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    return train_dataloader, eval_dataloader


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    """
    目的: モデルのパラメータを「weight decay あり」と「weight decay なし」に仕分ける。
    流れ: named_parameters() → パラメータ名で仕分け → 2 つのグループを作成して返す。
    なぜ: Transformer 系では LayerNorm や bias などに weight decay をかけない慣習があり、最適化を微調整するため。
    """
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


def evaluate_model(model, eval_dataloader, accelerator):
    """
    目的: 検証データを使ってモデルの損失・パープレキシティを計算し、評価する。
    流れ: eval mode → eval_dataloader ループ → forward 計算 (no_grad) → 損失を収集 → 平均 loss & perplexity を算出。
    ポイント: accelerator.gather による分散学習向け損失集計、最大ステップ制限の有無、計算結果を返す。
    """
    model.eval()
    losses = []

    for step, batch in enumerate(eval_dataloader):

        print(f"evaluate model step: {step}")

        with torch.no_grad():
            outputs = model(batch, labels=batch)
        loss = outputs.loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if args.max_eval_steps > 0 and step >= args.max_eval_steps:
            break

    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float("inf"))
    return loss.item(), perplexity.item()


def setup_logging(project_name, accelerator):
    """
    Python の logger だけを使用してロギングを行うためのセットアップ。
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # ログの整形や出力方法を指定
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # ファイル出力とコンソール出力の例
    file_handler = logging.FileHandler(
        f"{LOG_DIR}/debug_{accelerator.process_index}.log"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    if not accelerator.is_main_process:
        # サブプロセスで冗長なログを出さない
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    else:
        # メインプロセス
        datasets.utils.logging.set_verbosity_debug()
        transformers.utils.logging.set_verbosity_info()

    run_name = f"{project_name}_run"
    return logger, run_name


def log_metrics(step, metrics, logger, accelerator):
    """
    Python logger でのみログを出す関数。
    """
    logger.info(f"Step {step} | " + " | ".join(f"{k}: {v}" for k, v in metrics.items()))


def train_model(model, tokenizer):
    accelerator = Accelerator()
    samples_per_step = accelerator.state.num_processes * args.train_batch_size

    # ログのセットアップ
    logger, run_name = setup_logging(PROJECT_NAME, accelerator)
    logger.info(accelerator.state)

    # データローダー作成
    train_dataloader, eval_dataloader = create_dataloaders(tokenizer)

    # Optimizer & Scheduler
    optimizer = AdamW(get_grouped_params(model), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    def get_lr():
        return optimizer.param_groups[0]["lr"]

    # Accelerator で準備
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    model.train()
    completed_steps = 0

    for step, batch in enumerate(train_dataloader, start=1):
        outputs = model(batch, labels=batch)
        loss = outputs.loss

        # ログ出し (Python logger)
        log_metrics(
            step,
            {
                "lr": get_lr(),
                "samples": step * samples_per_step,
                "steps": completed_steps,
                "loss/train": loss.item(),
            },
            logger,
            accelerator,
        )

        # 勾配積算
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        if step % args.save_checkpoint_steps == 0:
            # チェックポイントで保存
            logger.info("Evaluating and saving model checkpoint")
            log_and_save(logger, model, eval_dataloader, accelerator, step)

            model.train()

        if completed_steps >= args.max_train_steps:
            break

    # 学習終了後に最終チェックポイント保存
    logger.info("Evaluating and saving model after training")
    log_and_save(logger, model, eval_dataloader, accelerator, step)


def log_and_save(logger, model, eval_dataloader, accelerator, step):
    """
    ログ出力、保存する関数。
    """
    # 評価
    eval_loss, perplexity = evaluate_model(model, eval_dataloader, accelerator)

    # ログ出力
    log_metrics(
        step,
        {"loss/eval": eval_loss, "perplexity": perplexity},
        logger,
        accelerator,
    )
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        # ローカルに保存
        ckpt_dir = f"{MODEL_SAVE_DIR}model_checkpoint_step_{step}"
        os.makedirs(ckpt_dir, exist_ok=True)
        unwrapped_model.save_pretrained(ckpt_dir)


# =====================================================================
# モデルの初期化
# =====================================================================
def init_model1(tokenizer_name: str, model_name: str):
    """
    プリトレーニング済みモデルを初期化してロードする
    """
    # 語彙サイズを合わせて初期化
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # tokenizer.model_max_length = 99999  # もしくは 1e30, 32768 など、十分大きい値
    config = AutoConfig.from_pretrained(model_name, vocab_size=len(tokenizer))
    model = AutoModelForCausalLM.from_config(config)

    return model, tokenizer


def init_model2(tokenizer_name: str, model_name: str):
    """
    プリトレーニング済みモデルをロードする
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
    )

    # プリトレーニング済みGPT-2をロード（トークナイザと同じvocab_sizeならconfig上書き不要）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )

    # トークナイザに独自トークンを追加しているなら
    # model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


# =====================================================================
# モデルの学習実行
# =====================================================================
def execute_train_model1():
    """
    モデルをリセットして学習する。
    """

    # モデルの初期化
    model, tokenizer = init_model1(
        tokenizer_name=TOKENIZER_NAME,
        model_name=MODEL_NAME,
    )

    # モデルの学習
    train_model(model, tokenizer)


def execute_train_model2():
    """
    モデルを追加で学習する。
    """

    # モデルの初期化
    model, tokenizer = init_model2(
        tokenizer_name=TOKENIZER_NAME,
        model_name=MODEL_NAME,
    )

    # モデルの学習
    train_model(model, tokenizer)


# =====================================================================
# モデルの実行
# =====================================================================
def execute_model(
    prompt: str,
    tokenizer_name: str = TOKENIZER_NAME,
    model_name: str = MODEL_NAME,
):
    """
    モデルの実行
    """
    # 1) トークナイザーをローカルから読み込み
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 2) モデル本体をローカルから読み込み
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 3) 推論時に必要ならデバイス指定（GPUがある場合）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 4) トークン化
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # 5) 推論（例: テキスト生成）
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=50,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9,
            top_k=50,
        )

    # 7) 出力結果のデコード
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)


if __name__ == "__main__":
    # デバッグ
    # from config.debug import *

    # 1. トークナイザーの学習＆保存
    new_tokenizer = train_tokenizer()

    # 2. モデルの学習＆保存
    # プリトレーニング済みモデルを初期化してロードする
    # execute_train_model1()

    # プリトレーニング済みモデルをロードする
    execute_train_model2()

    # 学習したモデルの実行
    execute_model(
        prompt="def main()",
        tokenizer_name=EXECUTE_TOKENIZER_DIR,
        model_name=EXECUTE_MODEL_DIR,
    )
