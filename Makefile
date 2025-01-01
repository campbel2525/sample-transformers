include .env.docker
pn := $(PROJECT_NAME)
user_name := $(USER_NAME)
user_group := $(USER_GROUP)

init: ## 開発環境作成
	make chown
	rm -rf code/.venv
	make destroy
	docker compose -p $(pn) build --no-cache
	docker compose -p $(pn) down --volumes
	docker compose -p $(pn) up -d
	docker compose -p $(pn) exec -it python pipenv install --dev
	make chown

up: ## 開発環境立ち上げ
	docker compose -p $(pn) up -d

down: ## 開発環境down
	docker compose -p $(pn) down

shell: ## dockerのshellに入る
	docker compose -p $(pn) exec python bash

check: ## コードのフォーマット
	docker compose -p $(pn) exec -it python pipenv run isort .
	docker compose -p $(pn) exec -it python pipenv run black .
	docker compose -p $(pn) exec -it python pipenv run flake8 .
	docker compose -p $(pn) exec -it python pipenv run mypy .
	make chown

destroy: ## 開発環境削除
	make down
	if [ -n "$(docker network ls -qf name=$(pn))" ]; then \
		docker network ls -qf name=$(pn) | xargs docker network rm; \
	fi
	if [ -n "$(docker container ls -a -qf name=$(pn))" ]; then \
		docker container ls -a -qf name=$(pn) | xargs docker container rm; \
	fi
	if [ -n "$(docker volume ls -qf name=$(pn))" ]; then \
		docker volume ls -qf name=$(pn) | xargs docker volume rm; \
	fi

push:
	git add .
	git commit -m "Commit at $$(date +'%Y-%m-%d %H:%M:%S')"
	git push origin main

# すべてのファイルの所有者を指定したユーザーに変更する
# .env.dockerのUSER_NAMEが指定されている場合に実行
chown:
	if [ -n "${user_name}" ]; then \
		sudo chown -R "${user_name}:${user_group}" ./ ; \
	fi

reset-commit: ## mainブランチのコミット履歴を1つにする 使用は控える
	git checkout --orphan new-branch-name
	git add .
	git branch -D main
	git branch -m main
	git commit -m "first commit"
	git push origin -f main
