PIP_SRC='./pip_src'
pip install --src $PIP_SRC -r requirements.txt
python -c "import nltk; nltk.download('punkt')"