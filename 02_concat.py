import pandas as pd
import glob

'''컬럼명 오류 수정하는법'''
# df_IT = pd.read_csv('./crawling/team_crawling/news_IT_1-50.csv', index_col=0)
# print(df_IT.columns)
'''df_IT.rename(columns={'수정할컬럼':'수정된컬럼'}, inplace=True)'''
# df_IT.rename(columns={'Title':'title'}, inplace=True)
# print(df_IT.columns)
# df_IT.to_csv('./crawling/team_crawling/news_IT_1-50.csv', index=False)


data_paths = glob.glob('./crawling/news_*.csv')
df = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path)
    df = pd.concat([df, df_temp])
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
print(df.tail())
print(df['category'].value_counts())
print(df.info())
df.to_csv('./crawling/naver_news.csv', index=False)