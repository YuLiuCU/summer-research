#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

class preprocess():

    def __init__(self,filepath):

        self.df=pd.read_csv(filepath)
        
        
        self.transform_date()
        self.raw_dataframe,self.trainning_dataframe,self.mean_dataframe,self.std_dataframe=self.preprocess()


    def transform_date(self):
        self.df["DATE"]=pd.to_datetime(self.df.DATE,format='%Y%m%d').dt.week


    def preprocess(self):

        raw_F,df_F,mean_F,std_F=self.preprocess_individual(self.df[self.df.SYM_ROOT=="F"])
        return  raw_F.reset_index(drop=True),df_F.reset_index(drop=True),mean_F.reset_index(drop=True),std_F.reset_index(drop=True)

    def preprocess_individual(self,df):
        df.drop(['TIME_M','SYM_ROOT','SYM_SUFFIX'],axis=1,inplace=True)
        mean=df.groupby('DATE').mean().iloc[:-1]
        std=df.groupby('DATE').std().iloc[:-1]
        mean.rename(columns={'BID':'BID_mean', 'BIDSIZ':'BIDSIZ_mean','ASK': 'ASK_mean','ASKSIZ': 'ASKSIZ_mean'},inplace=True)
        mean.index=mean.index+1
        std.index=std.index+1
        std.rename(columns={'BID':'BID_std', 'BIDSIZ':'BIDSIZ_std','ASK': 'ASK_std','ASKSIZ': 'ASKSIZ_std'},inplace=True)
        df.set_index(['DATE'],drop=True,inplace=True)
        mean=mean.join(std)
        df=df.join(mean)
        df=df.dropna()
        mean_df=self.create_40_features(df[['ASK_mean','ASKSIZ_mean','BID_mean', 'BIDSIZ_mean']],sorting=False)
        std_df=self.create_40_features(df[['ASK_std','ASKSIZ_std','BID_std', 'BIDSIZ_std']],sorting=False)
        raw_df = self.create_40_features(df[['ASK', 'ASKSIZ','BID', 'BIDSIZ']])
        df_final=raw_df-mean_df
        df_final/=std_df
        return raw_df,df_final,mean_df,std_df


    def create_40_features(self,df_final,sorting=True):

        new_df2=self.sorting(df_final.iloc[:10],sorting).values.reshape(1,40).tolist()
        for i in range(10,df_final.shape[0]-10,10):
            try:
                new_df2.extend(self.sorting(df_final.iloc[i:i+10],sorting).values.reshape(1,40))
            except:
                continue
        return pd.DataFrame(new_df2)  



    def sorting(self,df,sorting=True):
        if len(df.index.unique())==1:
            if sorting:
                frame=df[["BID","BIDSIZ"]]
                frame=frame.sort_values(by='BID',ascending=False).reset_index(drop=True)
                frame2=df[['ASK', 'ASKSIZ']]
                frame2=frame2.sort_values(by='ASK').reset_index(drop=True)
                frame2=pd.concat([frame2,frame],axis=1)
                return frame2
            else:
                return df
        else:
            raise ValueError()




data_prepared=preprocess('test.csv')
data_prepared.raw_dataframe.to_csv('test_raw.csv')
data_prepared.trainning_dataframe.to_csv('test_data.csv')
data_prepared.std_dataframe.to_csv('test_std.csv')
data_prepared.mean_dataframe.to_csv('test_mean.csv')

data_prepared=preprocess('val.csv')
data_prepared.raw_dataframe.to_csv('val_raw.csv')
data_prepared.trainning_dataframe.to_csv('val_data.csv')
data_prepared.std_dataframe.to_csv('val_std.csv')
data_prepared.mean_dataframe.to_csv('val_mean.csv')

data_prepared=preprocess('train.csv')
data_prepared.raw_dataframe.to_csv('train_raw.csv')
data_prepared.trainning_dataframe.to_csv('train_data.csv')
data_prepared.std_dataframe.to_csv('train_std.csv')
data_prepared.mean_dataframe.to_csv('train_mean.csv')