import os
import pandas as pd

pos_test_files = os.listdir("C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 3/data/aclImdb/test/pos")
neg_test_files = os.listdir("C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 3/data/aclImdb/test/neg")
pos_train_files = os.listdir("C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 3/data/aclImdb/train/pos")
neg_train_files = os.listdir("C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 3/data/aclImdb/train/neg")

train_pos_list = []
for file in pos_train_files:
    with open("C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 3/data/aclImdb/train/pos/{}".format(file), 'r', encoding='utf-8') as fd:
        text = fd.read()
        train_pos_list.append(text)

train_neg_list = []
for file in neg_train_files:
    with open("C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 3/data/aclImdb/train/neg/{}".format(file), 'r',encoding='utf-8') as fd:
        text = fd.read()
        train_neg_list.append(text)

train_data_pos = {'label' : 1, 'text': train_pos_list}
train_data_neg = {'label' :0, 'text': train_neg_list}
train_data_frame_complete = pd.DataFrame(train_data_pos).append(pd.DataFrame(train_data_neg)).reset_index(drop=True)
train_data_frame_complete.to_csv('imdb_train.csv', index=False)

test_pos_list = []
for file in pos_test_files:
    with open("C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 3/data/aclImdb/test/pos/{}".format(file), 'r', encoding='utf-8') as fd:
        text = fd.read()
        test_pos_list.append(text)

test_neg_list = []
for file in neg_test_files:
    with open("C:/Users/Yme/Documents/MEGA/Master DSE/Data Mining/Assignments/Assignment 3/data/aclImdb/test/neg/{}".format(file), 'r',encoding='utf-8') as fd:
        text = fd.read()
        test_neg_list.append(text)

test_data_pos = {'label' : 1, 'text': test_pos_list}
test_data_neg = {'label' :0, 'text': test_neg_list}
test_data_frame_complete = pd.DataFrame(test_data_pos).append(pd.DataFrame(test_data_neg)).reset_index(drop=True)
test_data_frame_complete.to_csv('imdb_test.csv', index=False)

df_imdb_train = pd.read_csv("imdb_train.csv")
df_imdb_test = pd.read_csv("imdb_test.csv")
imdb_final = df_imdb_train.append(df_imdb_test, ignore_index=True)
imdb_final.to_csv("imdb_final.csv", index=False)

df = pd.read_csv("imdb_final.csv")
print(df[0:100])