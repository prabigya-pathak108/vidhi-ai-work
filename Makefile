SHELL:=/usr/bin/env bash

# Load variables from .env for use in commands
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

DOCKER_COMPOSE := docker compose
APP_NAME := vidhi_ai

.DEFAULT_GOAL := help

##@ General
.PHONY: help
help: ## Display this help screen
	@echo "Usage: make [target]"
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1mTargets:\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Building
.PHONY: build
build: ## Build the docker images
	$(DOCKER_COMPOSE) build

##@ Start/Stop/Restart
.PHONY: start stop restart status

start: ## Start all services (detached)
	$(DOCKER_COMPOSE) up -d

stop: ## Stop all services
	$(DOCKER_COMPOSE) down

restart: ## Restart the app service
	$(DOCKER_COMPOSE) restart app

status: ## Show status of containers
	$(DOCKER_COMPOSE) ps

##@ Logging
.PHONY: logs
logs: ## Tail logs for the app service
	$(DOCKER_COMPOSE) logs -f app

##@ Shell Access
.PHONY: bash-app bash-db
bash-app: ## Enter the FastAPI container shell
	$(DOCKER_COMPOSE) exec app /bin/bash

bash-db: ## Enter the Postgres CLI inside the container
	$(DOCKER_COMPOSE) exec db psql -U $(POSTGRES_USER) -d $(POSTGRES_DB)

##@ Database Management
.PHONY: restart-pg
restart-pg: ## WIPE all records (Truncate) - Leaves tables intact
	@echo "⚠️ Warning: This will delete all users and chat history."
	@read -p "Are you sure? [y/N] " ans && if [ $${ans:-N} = y ]; then \
		$(DOCKER_COMPOSE) exec db psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "TRUNCATE users, chat_sessions, chat_messages CASCADE;"; \
		echo "✅ Database records cleared."; \
	fi

##@ Cleanup
.PHONY: prune
prune: ## Hard Reset: Remove containers, project-specific volumes and images (No confirmation)
	@echo "Stopping project and wiping local database volumes..."
	@$(DOCKER_COMPOSE) down -v --rmi local --remove-orphans
	@echo "✅ Project resources and volumes have been removed."

##@ Environment
.PHONY: setup
setup: ## Create .env from .env.example
	@cp .env.example .env && echo "✅ .env created. Update it with your secrets."

##@ Testing
.PHONY: test

test: ## Run tests with pytest
	@echo "🧪 Running tests..."
	@pytest tests/ -v

# Code quality
.PHONY: lint format

lint:
	@echo "🔍 Running linters..."
	flake8 src --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format:
	@echo "🎨 Formatting code..."
	black src/
	isort src/
	@echo "✅ Code formatted"

# Development
.PHONY: install dev clean

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

dev:
	@echo "🚀 Starting development server..."
	python main.py

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
	@echo "✅ Cleaned"