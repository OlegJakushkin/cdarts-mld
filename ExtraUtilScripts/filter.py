import random as r
import os
import pandas as pd
import time
from gensim.summarization import keywords
from gpt2_client import GPT2Client
import gpt_2_simple as gpt2
import swifter
def parse_csv():
    df = pd.read_csv("./train.csv")
    print(df.head(5))
    def kwp(text):
        line = ', '.join(keywords(text, words=13).split('\n'))
        return line

    df['keywords'] = df['abstract'].swifter.apply()
    df.to_csv("parsed.csv")

def generate_text_corpus():
    df = pd.read_csv("./parsed.csv")
    print(df.head(5))
    text = ''
    i = 0

    for index, row in df.iterrows():
        line = ''
        line  += "<KEYS:> " + row['keywords']
        line += "; <TITLE:> " + row['title'] + " <END>"
        line  += "\n"
        text += line;
        i = i + 1
        if i % 1000 == 0:
            print("stage 2: " + str(i))
            print("sample: " + line)
    outF = open("parsed-train.txt", "w")
    outF.write(text)
    outF.close()


class GPT2EC(GPT2Client):
    def __init__(self,  **kwargs):
        GPT2Client.__init__(self, **kwargs)

    def finetune(self, corpus, steps=1000, return_text=True):
        sess = gpt2.start_tf_sess()
        gpt2.finetune(sess,
                corpus,
                model_name=self.model_name,
                steps=steps,
                multi_gpu=True)     # steps is max number of training steps

        if return_text:
            text = gpt2.generate(sess, return_as_list=True)
            return text
        else:
            gpt2.generate(sess)

    def load(self, sess):
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(sess)

        gpt2.generate(sess)

def train_on_corpus():
    gpt2c = GPT2EC('1558M', save_dir='models') # This could also be `345M`, `774M`, or `1558M`
    gpt2c.load_model()
    my_corpus = './parsed-train.txt' # path to corpus
    custom_text = gpt2c.finetune(my_corpus, 5000) # Load your custom dataset


parse_csv()
print("\n---\nparsed!\n---\n")
generate_text_corpus()
print("\n---\ngenerated corpys!\n---\n")
train_on_corpus()