import os
import urllib.request



os.system('git clone https://github.com/TheAtticusProject/cuad.git')

os.system('mv cuad cuad-training')
os.system('mv cuad cuad-training')
os.system('unzip cuad-training/data.zip -d cuad-data/')
os.system('mkdir cuad-models')
os.system('mkdir cuad-models')
os.system('unzip cuad-models/roberta-base.zip -d cuad-models/')
os.system('pip install torch')
os.system('pip install transformers')