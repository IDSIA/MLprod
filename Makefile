# This is an utility for simplify the execution of repetitive commands using docker-compose.
build:
	docker compose --env-file .env -f docker/docker-compose.yaml -p mlprod build

start:
	docker compose --env-file .env -f docker/docker-compose.yaml -p mlprod up -d

stop:
	docker compose --env-file .env -f docker/docker-compose.yaml -p mlprod down

reload:
	docker compose --env-file .env -f docker/docker-compose.yaml -p mlprod build
	docker compose --env-file .env -f docker/docker-compose.yaml -p mlprod up -d

grafana:
	docker compose --env-file .env -f docker/docker-compose.yaml -p mlprod stop grafana
	docker compose --env-file .env -f docker/docker-compose.yaml -p mlprod start grafana

doc:
	sphinx-build -b html docs/source/ docs/build/html
