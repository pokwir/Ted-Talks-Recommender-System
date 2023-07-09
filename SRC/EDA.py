import pandas as pd
df = pd.read_csv('../Data_output/talks_2.csv').drop(columns='Unnamed: 0')

df = df.dropna()
df = df.drop_duplicates()
# checking the duplicated transcript
# display(df[df['transcript'].duplicated(keep=False)])
# delete that record
df = df.drop(index=df[df['transcript'].duplicated(keep='last')].index)
# removing wrong values in likes and views
df = df[pd.to_numeric(df['likes'], errors='coerce').notna()]
df = df[pd.to_numeric(df['views'], errors='coerce').notna()]
# replacing special characters (e.g. $quot;) with proper syntax
df['transcript'] = df['transcript'].replace(['&quot;','&apos;', '&amp;', '&lt;', '&gt;'], 
                                            ["'","'"," and ", " less than ", " greater than "], regex=True)
# updating datatype                                            
df['likes'] = df['likes'].astype('int')
df['views'] = df['views'].astype('int')   
# optional save to file
# df.to_csv('../Data_output/ted_talk_clean.csv')                                         