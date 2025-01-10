# 概要

docker から nvidia の gpu を使用する前提です

windows 前提になっていますびので mac を使用する場合や gpu を使用しない場合は`docker-compose.yml`を適宜変更してください

# 開発環境の作成

## 1. 環境変数

`.env.docker.example`を参考に`.env.docker`を作成

`code/.env.example`を参考に`/code/.env`を作成

## 2 docker の構築

```
make init
```

# 実行方法

```
make shell
pipenv run python src/train_ai.py
```

# フォルダについて

- code
  - プログラム配置
- code/hf_repo/models
  - 学習したモデルを保存
- code/hf_repo/tokenizers
  - 学習したトークナイザーの配置
