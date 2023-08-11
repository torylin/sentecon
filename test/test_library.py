from context import SenteCon

sentecon = SenteCon(lexicon='LIWC', lm='all-mpnet-base-v2', liwc_path='/home/victorialin/Documents/liwc_dict/LIWC2015_English_Flat.dic')
df = sentecon.embed(['i am so happy', 'i am so unhappy'])

print(df)

sentecon2 = SenteCon(lexicon='Empath', lm='all-mpnet-base-v2')
df2 = sentecon2.embed(['i am so happy', 'i am so unhappy'])
print(df2)