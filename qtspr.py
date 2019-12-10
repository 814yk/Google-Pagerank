from scipy.sparse import csr_matrix
import numpy as np
import pagerank

def initp():
    #make list to build sparse transition matrix
    txt_ptspr = open('data/doc_topics.txt', 'r')
    row_ptspr = []
    col_ptspr = []
    data_ptspr = []
    for line in txt_ptspr:
        temp = line.split(' ')
        row_ptspr.append(int(temp[1]) - 1)
        col_ptspr.append(int(temp[0]) - 1)
        data_ptspr.append(1)
    dim_row = max(row_ptspr) + 1
    dim_col = max(col_ptspr) + 1
    p = csr_matrix((data_ptspr, (row_ptspr, col_ptspr)), shape=(dim_row, dim_col), dtype=np.float)
    return p

def qtspr_off():
    alpha=0.8
    beta=0.1
    gamma=0.1
    p=initp()
    M_,M_empty = pagerank.transition_matrix()# get matrix from pagerank
    M=M_.transpose()
    #M_empty=M_empty_.transpose()
    [row, col] = M_.shape
    topic_num = p.shape[0]
    off_vec = []
    for i in range(topic_num):
        p_ = np.transpose(p[i].toarray())
        p_t=(p_/p_.sum(axis=0)).squeeze() 
        #p_t=p_.squeeze()      when not dividing
        p_0 = np.divide(np.ones(row), row)
        type_max = 1 << np.finfo(np.float64).nmant
        r_t = np.random.randint(0, type_max, size=row) / np.float64(type_max)
        n=1
        while n < 1000:
            r_new=(alpha*M*r_t+beta*p_t+gamma*p_0)+alpha*(M_empty*r_t)[0]
            r_t=r_new
            if pow(r_t-r_new,2).sum(axis=0)**0.5 <pow(10,-309):
                break
            n+=1
        off_vec.append(r_t)
    return off_vec

def qtspr_on():
    # calculate online ptspr
    import time
    ptspr_on_start = time.time() 
    p=initp()
    row = p.shape[0]
    col = p.shape[1]
    topic = open("data/query-topic-distro.txt", 'r')
    QTSPR = []
    off_vec=qtspr_off()
    for line in topic:
        data = line.split(' ')
        online_q = np.empty((row, col))
        for i in range(2, len(data)):
            online_q[i-2] = off_vec[i-2] * float(data[i].split(':')[1])
        QTSPR.append(online_q.sum(axis=0))
    print("{} secs for qts_pagerank_online".format(round(time.time() - ptspr_on_start,4)),"/ round(time,4)")
    return QTSPR

def filewrite():
    QTSPR=qtspr_on()
    f=open('QTSPR-U2Q1.txt', 'w')
    n=0
    for i in QTSPR:
        n+=1
        f.write(str(n)+" "+str(i)+'\n')
        
if __name__ == "__main__":
    print("Calculating QTSPR.....")
    f=filewrite()
    print("Done!\n\n")
