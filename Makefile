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
	dcoker-compose stop grafana
	docker-compose start grafana
