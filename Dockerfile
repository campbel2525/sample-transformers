FROM nvcr.io/nvidia/pytorch:24.11-py3

# apt-getのアップデート
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    locales \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen ja_JP.UTF-8 \
    && update-locale LANG=ja_JP.UTF-8

# デフォルトのディレクトリを設定
# イメージにディレクトリがないので作成される
WORKDIR /project

# project配下に.venvを作成する
ENV PIPENV_VENV_IN_PROJECT=1

# log出力をリアルタイムにする
ENV PYTHONUNBUFFERED=1

# キャッシュを作成しない
ENV PYTHONDONTWRITEBYTECODE=1

# パスを通す
ENV PYTHONPATH="/project"

# pipのアップデート
RUN pip install --upgrade pip

# pipenvのインストール
RUN pip install --upgrade setuptools pipenv
