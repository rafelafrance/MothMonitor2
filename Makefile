.PHONY: test install dev clean
.ONESHELL:

test:
	uv run -m unittest discover

install:
	uv sync

dev:
	uv sync
	uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/rocm7.0

clean:
	rm -rf .venv
	rm -rf build
	rm -rf MothMonitor2.egg-info
	find -iname "*.pyc" -delete
