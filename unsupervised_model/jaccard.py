from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    
    # �����м����ո�
    s1, s2 = add_space(s1), add_space(s2)
    # ת��ΪTF����
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # �󽻼�
    numerator = np.sum(np.min(vectors, axis=0))
    # �󲢼�
    denominator = np.sum(np.max(vectors, axis=0))
    # ����ܿ���ϵ��
    return 1.0 * numerator / denominator


s1 = '���ڸ�����'
s2 = '���ڸ�ʲô��'
print(jaccard_similarity(s1, s2))
