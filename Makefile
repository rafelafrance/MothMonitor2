.PHONY: test install dev clean
.ONESHELL:

test:
	uv run -m unittest discover

install:
	uv sync

clean:
	rm -rf .venv
	rm -rf build
	rm -rf MothMonitor2.egg-info
	find -iname "*.pyc" -delete
