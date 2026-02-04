.PHONY: test install dev clean
.ONESHELL:

test:
	uv run -m unittest discover

install:
	uv sync

dev:
	uv sync
	uv pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1

clean:
	rm -rf .venv
	rm -rf build
	rm -rf MothMonitor2.egg-info
	find -iname "*.pyc" -delete
