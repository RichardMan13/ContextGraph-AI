.PHONY: help clean lint requirements up down enrich run evaluate

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = cinegraph-ai
PYTHON       = .venv/Scripts/python
UVICORN      = .venv/Scripts/uvicorn

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
requirements:
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install -r requirements.txt

## Start Docker services (PostgreSQL + AGE + pgvector, MLflow)
up:
	docker-compose up -d
	docker-compose ps

## Stop and remove Docker containers (keeps named volumes)
down:
	docker-compose down

## Stop Docker and DESTROY all volumes (full reset)
down-v:
	docker-compose down -v

## Enrich movies.csv with OMDb plot descriptions → data/interim/movies_enriched.csv
enrich:
	$(PYTHON) src/data/enrich_plots.py

## Dry-run enrichment (validates setup without consuming API quota)
enrich-dry:
	$(PYTHON) src/data/enrich_plots.py --dry-run

## Ingest movies_enriched.csv into Apache AGE knowledge graph
ingest:
	$(PYTHON) src/data/ingest_graph.py

## Dry-run ingestion (validates setup without writing to the database)
ingest-dry:
	$(PYTHON) src/data/ingest_graph.py --dry-run

## Generate OpenAI embeddings for movie plots and store in pgvector
embed:
	$(PYTHON) src/data/generate_embeddings.py

## Dry-run embeddings (validates data logic without API hits)
embed-dry:
	$(PYTHON) src/data/generate_embeddings.py --dry-run

## Start FastAPI + Gradio server (http://localhost:7860)
run:
	$(UVICORN) src.app:app --host 0.0.0.0 --port 7860 --reload

## Run RAGas evaluation and log results to MLflow
evaluate:
	$(PYTHON) src/models/evaluate.py

## Lint source code with flake8
lint:
	.venv/Scripts/flake8 src

## Delete compiled Python files and caches
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete

## Test Python environment
test_environment:
	$(PYTHON) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
