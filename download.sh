# write cookie data into cookies.txt
# https://www.kaggle.com/c/facial-keypoints-detection/data

# download data files and save them under /data
cd data
wget -x --load-cookies ./../cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip
wget -x --load-cookies ./../cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip
wget -x --load-cookies ./../cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/SampleSubmission.csv
wget -x --load-cookies ./../cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/IdLookupTable.csv
mv www.kaggle.com/c/facial-keypoints-detection/download/* .
sudo apt-get install unzip
unzip training.zip
unzip test.zip
rm -rf www.kaggle.com/
