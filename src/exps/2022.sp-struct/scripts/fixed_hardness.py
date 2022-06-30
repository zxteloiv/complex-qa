

import numpy as np
# import matplotlib.pyplot as plt
from typing import List
import re
import json
import numpy as np

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit')
AGG_OPS = ('max', 'min', 'count', 'sum', 'avg')

def normalize_sql(sql: str):
    sql = sql.replace('(',' ( ')
    sql = sql.replace(')',' ) ')
    return sql.lower()

def eval_score(sql: List[str]):
    
    scores = []
    for idx, word in enumerate(sql):
        if word in CLAUSE_KEYWORDS:
            
            layer = sql[0:idx].count('(') - sql[0:idx].count(')') + 1
            if layer > 5:
                layer = 5
            next_key_word_index = idx + 1

            while not sql[next_key_word_index] in CLAUSE_KEYWORDS:
                next_key_word_index += 1
                if next_key_word_index > len(sql)-1:
                    break

            if not word == 'where':
                count = sql[idx:next_key_word_index].count(',') + 1
            else:
                count = 1
                stack = []
                for word1 in sql[idx:]:
                    if word1 == '(':
                        stack.append(word1)
                    if word1 == 'select':
                        stack.append(word1)
                    if word1 == ')':
                        if len(stack) == 0:
                            break
                        else:
                            if stack[-1] == 'select':
                                stack.pop()
                                stack.pop()
                            elif stack[-1] == '(':
                                stack.pop()
                    if (word1 == 'and' or word1 == 'or') and len(stack) == 0:
                        count += 1
                

            if count > 1:
                score = 2
            else:
                score = 1
            scores.append(layer * score)
    
    return sum(scores)


def count_component1(sql: str) -> int:
    count = 0
    
    if sql.count('WHERE') > 0:
        count += 1
    if sql.count('GROUP BY') > 0:
        count += 1
    if sql.count('ORDER BY') > 0:
        count += 1

    #from后的表个数（从from到下一个关键字之间的逗号数）
    from_index= sql.find('FROM')
    next_key_word_index = min([sql.find(key_word, from_index) for key_word in CLAUSE_KEYWORDS])
        

    comma_count = sql.count(',', from_index, next_key_word_index)
    count += comma_count

    count += sql.count('or') #or的个数
    count += sql.count('like') #like的个数

    return count


def count_nest(sql: str) -> int:
    return sql.count('SELECT') - 1 #用select个数近似嵌套数


def count_others(sql: str) -> int:
    count = 0

    #聚合关键字的出现次数
    agg_count = sum([sql.count(agg_ops) for agg_ops in AGG_OPS])
    count += agg_count

    #select后列的是否大于1(从select到下一个关键字之间是否有逗号)
    select_index = sql.find('SELECT')
    next_key_word_index = min([sql.find(key_word, select_index) for key_word in CLAUSE_KEYWORDS])
    
    
    if sql.count(',', select_index, next_key_word_index) > 0:
        count += 1

    #where后是否有多个条件(从where到下一个关键字之间是否有and/or)
    where_index = sql.find('WHERE')
    if not where_index == -1:
        next_key_word_index = min([sql.find(key_word, where_index) for key_word in CLAUSE_KEYWORDS])
    
        
        if sql.count('and', where_index, next_key_word_index) > 0 or sql.count('or', where_index, next_key_word_index) > 0:
            count += 1

    #group by后的列数是否大于1(从group by到下一个关键字之间是否有逗号)
    groupby_index = sql.rfind('GROUP BY')
    if not groupby_index == -1:
        next_key_word_index =  min([sql.find(key_word, groupby_index) for key_word in CLAUSE_KEYWORDS])

        
        if sql.count(',', groupby_index, next_key_word_index) > 0:
            count += 1

    return count



class Evaluator:
    def __init__(self) -> None:
        pass

    def eval_hardness(self, sql) -> str:
        sql = ' '.join(sql.lower().split())
        for key_word in CLAUSE_KEYWORDS:
            sql = sql.replace(key_word,key_word.upper())
        # print(sql)
        count_comp1_ = count_component1(sql)
        count_nest_ = count_nest(sql)
        count_others_ = count_others(sql)

        # print(count_comp1_, count_nest_, count_others_)
        if count_comp1_ <= 1 and count_others_ == 0 and count_nest_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_nest_ == 0) or \
                (count_comp1_ <= 2 and count_others_ < 2 and count_nest_ == 0):
            return "medium"
        elif (count_others_ > 2 and count_comp1_ <= 2 and count_nest_ == 0) or \
                (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_nest_ == 0) or \
                (count_comp1_ <= 1 and count_others_ == 0 and count_nest_ <= 1):
            return "hard"
        else:
            return "extra"


def file_stat(filename):
    f = open(filename)
    json_obj = json.load(f)
    f.close()

    all_scores = []

    for item in json_obj:
        sql = item['sql'][0]
        sql = normalize_sql(sql)
        score = eval_score(sql.lower().split())
        # all_scores.append(score)
        all_scores.extend([score] * len([sent['text'] for sent in item['sentences'] if sent['question-split'] != 'exclude']))

    print(filename)
    print(len(all_scores))
    # print(all_scores)
    print(np.mean(all_scores), np.std(all_scores))
    print(np.quantile(all_scores, [.5, .75, .9, 1]))


def main():
    import sys
    # f = open('scholar/final_test.json')
    train='/Users/zxarukas/code/complex-qa/data/CompGen/sql data/atis/schema_full_split/aligned_train.json'
    file_stat(train)
    test='/Users/zxarukas/code/complex-qa/data/CompGen/sql data/atis/schema_full_split/final_test.json'
    file_stat(test)

    train='/Users/zxarukas/code/complex-qa/data/CompGen/sql data/advising/schema_full_split/aligned_train.json'
    file_stat(train)
    test='/Users/zxarukas/code/complex-qa/data/CompGen/sql data/advising/schema_full_split/final_test.json'
    file_stat(test)

    train='/Users/zxarukas/code/complex-qa/data/CompGen/sql data/geography/schema_full_split/aligned_train.json'
    file_stat(train)
    test='/Users/zxarukas/code/complex-qa/data/CompGen/sql data/geography/schema_full_split/final_test.json'
    file_stat(test)

    train='/Users/zxarukas/code/complex-qa/data/CompGen/sql data/scholar/schema_full_split/aligned_train.json'
    file_stat(train)
    test='/Users/zxarukas/code/complex-qa/data/CompGen/sql data/scholar/schema_full_split/final_test.json'
    file_stat(test)


def foo():
    print(eval_score(normalize_sql(sql).split()))

if __name__ == '__main__':
    main()

# data = np.array(all_scores)
# plt.hist(data, bins=40)
# plt.title('scholar')
# plt.savefig('./scholar.jpg')
# plt.show()


