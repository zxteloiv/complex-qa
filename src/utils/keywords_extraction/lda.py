from gensim.models.ldamodel import LdaModel
import numpy as np

class topic_model():
    """docstring for topic_model"""
    def __init__(self,path='523model/523.model'):
        self.lda=LdaModel.load(path)
        self.dictionary=self.lda.id2word

    def get_doc_topics(self,text):
        text=text.split()
        self.bow=self.dictionary.doc2bow(text)
        return np.array(self.lda.get_document_topics(self.bow,0))[:,1]

    def get_doc_keywords(self,text,num=3):
        #out=[]
        out={}
        topics = self.get_doc_topics(text)
        topic = topics.reshape(1,len(topics))
        words_matrix = np.zeros((len(topics),len(self.bow)))
        if len(self.bow)==0:
            return out

        for n in range(0,len(self.bow)):
            word=self.bow[n]
            word_vec=self.lda.get_term_topics(word[0],0)
            for t in word_vec:
                words_matrix[t[0]][n]=t[1]
        weight=np.dot(topic,words_matrix)
        for i in range(num):
            max=0
            for j in range(0,np.size(weight[0])):
                if weight[0][j]>weight[0][max]:
                    max=j
            if weight[0][max]==0.0:
                break
            #out.append((self.dictionary[self.bow[max][0]],weight[0][max]))
            out[self.dictionary[self.bow[max][0]]]=weight[0][max]
            weight[0][max]=0
        return out

if __name__ == '__main__':
    # mo=topic_model('521model/521.model')#可选项为加载模型的路径
    # #mo.lda.save('533.model')
    # #print(mo.get_doc_topics('对于 国内 动漫 画作者 引用 工笔 素材 的 一些 个人 意见 。'))#输入的文本必须是已分词，且用空格隔开的字符串
    # print(mo.get_doc_keywords('你 好 ，不是 钢铁 怎样 炼成 的 而是 静静的 顿河',10))#输入文本同上，可选项为要选的关键词的数量，如果超过词袋数目，则不返回关键词
    # print(mo.get_doc_keywords('八路军 乔装 改 扮成 日本 军官 刺探 军情',10))#输入文本同上，可选项为要选的关键词的数量，如果超过词袋数目，则不返回关键词
    # print(mo.get_doc_keywords('我 要 是 以后 再 下午 喝 咖啡 我 他妈 就是  🍆 。',10))#输入文本同上，可选项为要选的关键词的数量，如果超过词袋数目，则不返回关键词
    # print(mo.get_doc_keywords('献给 所有 和 奶奶 有 着 童年 的 孩子 。',10))#输入文本同上，可选项为要选的关键词的数量，如果超过词袋数目，则不返回关键词

    import sys
    import os.path
    import jieba

    tm = topic_model(os.path.expanduser('~/.complex_qa/topic_model/topic.model'))
    for line in open(sys.argv[1]):
        sents = line.rstrip().split('\t')
        topics = []
        for sent in sents:
            text = ' '.join(jieba.cut(sent))
            kwds = tm.get_doc_keywords(text, 3)
            topics.extend(kwds.keys())
        topics = list(set(topics))
        print('\t'.join(sents + [','.join(topics)]))
