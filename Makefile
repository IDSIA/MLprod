# This is an utility for simplify the execution of repetitive commands using docker-compose.
build:
	docker-compose build

start:
	docker-compose up -d

stop:
	docker-compose down

reload:
	docker-compose build
	docker-compose up -d

grafana:
	docker-compose stop grafana
	docker-compose start grafana

doc:
	sphinx-build -b html docs/source/ docs/build/html
