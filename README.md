# cross_nlp
paper:A Cognition Based Attention Model for Sentiment Analysis
## requirement
python 3  
nltk  
textstat  

dataframe data can be accessed in 
链接：https://pan.baidu.com/s/1oCXJEjCzOc1HuOn8ONzdhA 
提取码：3ecs 
复制这段内容后打开百度网盘手机App，操作更方便哦  
original Mishra data address  
http://www.cfilt.iitb.ac.in/~cognitive-nlp/resources/Eye-tracking_and_SA-II_released_dataset.zip

## experiments
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