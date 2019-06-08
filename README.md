# cross_nlp
paper:A Cognition Based Attention Model for Sentiment Analysis
## requirement
python 3  
nltk  
textstat  
pytorch  
spacy

dataframe data can be accessed in 
链接：https://pan.baidu.com/s/1oCXJEjCzOc1HuOn8ONzdhA 
提取码：3ecs 
复制这段内容后打开百度网盘手机App，操作更方便哦  
original Mishra data address  
http://www.cfilt.iitb.ac.in/~cognitive-nlp/resources/Eye-tracking_and_SA-II_released_dataset.zip

## experiments
experiments repair:  
Download data frame data from Baidu Cloud, creat a "data" directory in root and put the data there.  
Or If You Want to Generate Features Dataframe Files yourself:  
1. Download raw data and run utils/convert_data_df.py and utils/convert_Mishra_data.py
2. Run cal_feature_thread.py and cal_feature_thread_Mishra.py  
PS: My code to calculate features are too slow and need about 4 hours on my server. I would be glad if some one can speed up my
code.
     
regression1_3:  
cross validation for all 14 people data  
regression1_4:  
cross validation for each person  
regression1_5:  
repeat 1_3, but use the percentage of reading time for each person for regression target  
regression1_6:  
repeat 1_3, but nomalize the features first
regression1_7:  
use RNN to regress 
regression1_8:  
Split data beased on different sentences.  
regression2_0:  
cross validation for Mishra dataset.