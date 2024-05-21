PIP_SRC='./pip_src'
pip install --src $PIP_SRC -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
git clone https://github.com/VarunGumma/IndicTransTokenizer pip_src/IndicTransTokenizer
pip install -e pip_src/IndicTransTokenizer