import pickle


def show_idf_result(doc=[],num=3):
    out={}
    for n in range(0, num):
        max=0
        for N in range(0, len(doc)):
            if doc[N][1] > doc[max][1]:
                max = N
        if doc[max][1]==0.0:
            break
        out[doc[max][0]]=doc[max][1]
        doc[max][1]=0.0
    return out

class tfidf_model():
    def __init__(self,model=None):
        if model!=None:
            self.load(model)
    def load(self,model):
        a=pickle.load(open(model,'rb'))
        self.tfidf=a.tfidf
        self.dictionary=a.dictionary
        self.id2word=a.id2word
    def decode_idf(self,text):
        doc=[list(word) for word in text]
        for word in doc:
            word[0] = self.id2word[word[0]]
        return doc
    def get_doc_keywords(self, text, topn=4):
        text=text.split()
        text=self.dictionary.doc2bow(text)
        return show_idf_result(self.decode_idf(self.tfidf[text]),topn)

if __name__ == '__main__':
	mo=tfidf_model('tfidf.pkl')
	print(mo.get_doc_keywords('没有 我 妈妈 就 在 家里',3))
	print(mo.get_doc_keywords('八路军 乔装 改 扮成 日本 军官 刺探 军情',3))
