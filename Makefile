.PHONY: wheel install test clean

wheel:
	python3 -m pip wheel . --no-deps -w dist

install:
	python3 -m pip install dist/*.whl

test:
	python3 -m pytest -q

clean:
	rm -rf dist build src/*.egg-info *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
