filter:
	python scripts/run_filter.py --config configs/default.yaml

test:
	pytest tests/ -v

.PHONY: filter test

