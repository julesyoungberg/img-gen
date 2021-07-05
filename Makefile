lint:
	poetry run pylint img_gen

format:
	poetry run black img_gen tests
