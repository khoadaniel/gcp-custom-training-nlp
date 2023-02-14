# A framework to create custom training job in GCP
### Credits to: Sascha Heyer

## Overview
This repository contains the necessary files to create custom training jobs in GCP for an NLP model. Here's a brief description of the files:

- `train.py`: this is the training script that takes input from a GS bucket and saves the trained model in another bucket.
- `Dockerfile`: is used to build a Docker image that contains the necessary dependencies and configuration to run the train.py script. It defines the environment in which the script will run, including the base image, any additional libraries needed, and any environment variables that need to be set.
- `cloudbuild.yaml`: this file contains the Cloud Build script that builds the container and registers it to the GCP Container Registry.
- `config.yaml`: this file contains the configurations for the machine that will run the training job. It is used by the `gcloud ai custom-jobs create` command.
- `requirements.txt`: this file contains the Python dependencies that are required by the training script.


## Usage
To use this repository, you'll need to follow these steps:

1. Clone the repository to your local machine or Google Colab.
- In case of Google Colab, here is the basic set-up steps:
```
from google.colab import auth
auth.authenticate_user()
```

```
!gcloud config set project ml-engineer-playground
```
Note: ml-engineer-playground -> is your GCP working project.

2. Update the `train.py` script to fit your specific use case.
3. Update the `config.yaml` file to configure the machine that will run the training job.
4. Build the Docker container the `cloudbuild.yaml` script.
```
!gcloud builds submit  --config cloudbuild.yaml
```
This  `cloudbuild.yaml` will trigger the `Dockerfile` and build the image, then push / register the container to the GCP Container Registry.

6. Run the training job using the `gcloud ai custom-jobs create` command, passing in the `config.yaml` file.
- Some variables needed to spin up the training job:
```
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
JOB_NAME=f"my_nlp_bert_{TIMESTAMP}"
CUSTOM_CONTAINER_IMAGE_URI="gcr.io/ml-engineer-playground/nlp-my-image"
REGION="europe-west1"
```
```
!gcloud ai custom-jobs create \
  --region={REGION} \
  --display-name={JOB_NAME} \
  --config=config.yaml
```

That's it! You now have a fully functional NLP training pipeline that you can use to train your (custom) models in GCP.



## Remarks
In GCR, an image name has the format `gcr.io/{project-id}/{image-name}:{tag}`. In this case, `gcr.io` specifies the registry, `ml-engineer-playground` is the GCP project ID, `nlp-my-image` is the name of the image, and there is no tag specified (so the default tag "latest" will be used).

So, in short, `gcr.io/ml-engineer-playground/nlp-my-imagw" is a specific path to an image in GCR. It's not a completely arbitrary name.



## Reference & Credits
https://www.youtube.com/watch?v=GM-tibVly_A

