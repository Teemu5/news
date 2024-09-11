on wsl create venv:
python3 -m venv tf 
pip tensorflow[and-cuda] install datasets recommenders  nltk transformers notebook
to run:
source tf/bin/activate    
jupyter notebook --no-browser --ip=0.0.0.0

Check for tf installation https://www.tensorflow.org/install/pip#linux_1

(To upload datasets from kaggle there needs to be kaggle api token at model/kaggle.json)