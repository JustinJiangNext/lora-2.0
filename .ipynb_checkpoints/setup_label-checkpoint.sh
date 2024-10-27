python3 -m pip install gpustat
python3 -m pip install --upgrade pip
python3 -m pip install transformers jsonlines accelerate torch 
export HF_HOME=MODELS

mkdir tmpbin
cd tmpbin 
curl -O https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar 
tar xf fivek_dataset.tar 
cd ..
