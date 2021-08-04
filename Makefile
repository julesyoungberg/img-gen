PROJECT_ID=img-gen-319216
IMAGE_NAME=img-gen-cycle-gan
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_NAME}

.PHONY: login lint format build train push

login:
	gcloud auth activate-service-account --key-file=./service_account.json

lint:
	poetry run pylint img_gen

format:
	poetry run black img_gen tests

build:
	docker build -t ${IMAGE_URI} .

push:
	docker push ${IMAGE_URI}

# TODO take job name from command line
train:
	gcloud ai-platform jobs submit training img_gen_cycle_can_training_job_1 --region us-west1 --master-image-uri ${IMAGE_URI} --scale-tier custom --master-machine-type e2-highmem-4
