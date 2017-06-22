import preprocessing as pre

df, labels = pre.read_data()
clean, words = pre.clean_data(df.iloc[:10])