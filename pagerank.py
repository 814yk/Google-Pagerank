import scipy as sc
from scipy.sparse import csr_matrix
import numpy as np
#from tqdm import trange,tqdm_notebook
import time

def transition_matrix():
    txt_trans = open( "data/transition.txt", 'r')
    #make list to build sparse transition matrix
    row_trans=[]
    col_trans=[]
    data_trans=[]
    dic_N={}
    for line in txt_trans:
        temp = line.split(' ')
        row_trans.append(int(temp[0])-1)
        col_trans.append(int(temp[1])-1)
        #making dictionary to gather row data information
        # try&except is faster than items, get in my computer
        try:
            count=dic_N[int(temp[0])-1]
            count+=1
            dic_N[int(temp[0])-1]=count
        except KeyError as e:
            count=1
            dic_N[int(temp[0])-1]=count
    txt_trans.close()
    for _,j in enumerate(row_trans):
        data_trans.append(float(1/dic_N[j]))
    dim=max(max(row_trans),max(col_trans))+1
    #make list contain non-zero rows information
    row_trans_=set(row_trans)
    row_zero=[]
    for i in range(dim):
        if i not in row_trans_:
            row_zero.append(i)
    col_zero=[0]*len(row_zero)
    val=1/dim
    data_zero=[val]*len(row_zero)
    #row_zero=sorted(row_zero_*(dim))
    #col_zero=list(range(dim))*len(row_zero_)
    #a=1/dim
    #data_zero=[a]*len(col_zero)

    #fin_row_trans=row_trans+row_zero
    #fin_col_trans=col_trans+col_zero
    #fin_data_trans=data_trans+data_zero

    #make non ergodic transition matrix, It will be same with ergodic transition matrix by calculation trick
    mat_trans=csr_matrix((data_trans, (row_trans, col_trans)), shape=(dim,dim))
    #make simple 1,dim matrix to use calculation trick
    mat_0=csr_matrix((data_zero,(col_zero,row_zero)),shape=(1,dim))
    #mat_empty=csr_matrix((data_zero, (row_zero, col_zero)), shape=(dim,dim))
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    return mat_trans,mat_0


def pagerank():
    import time
    pagerank_start = time.time() 
    a = 0.8
    mat_trans_,mat_0=transition_matrix()
    dim,_=mat_trans_.shape
    p0 = np.ones(dim)/(dim)
    mat_trans=mat_trans_.transpose()
    #mat_empty=mat_empty_.transpose()
    pr = np.ones(dim+1).reshape(-1)
    type_max = 1 << np.finfo(np.float64).nmant
    pr = np.random.randint(0, type_max, size=dim) / np.float64(type_max)
    #pr=np.random.dirichlet(dim,size=(dim,))
    #pr=np.ones(dim+1).transpose()
    #pr=pr.fill(float(1.0/(dim+1)))
    temp=0
    #while pow(pr-temp,2).sum(axis=0)**0.5 >pow(10,-309):
    for i in range(50000):
        #this is calculation trick, add constant (1-a)*(mat_0*pr)[0] to point vector
        temp =(a*(mat_trans*pr)+(1-a)*p0)+a*(mat_0*pr)[0]
        
        if pow(pr-temp,2).sum(axis=0)**0.5 <pow(10,-309):
            break
        pr = temp
    print("{} secs for pagerank ".format(round(time.time() - pagerank_start,4)),"/ round(time,4)")
    return pr


def filewrite():
    f = open('GPR.txt', 'w')
    doc_id = 0
    pr=pagerank()
    for i in pr:
        doc_id += 1
        f.write(str(doc_id) + " " + str(i) + '\n')
    f.close()
    
def loadgpr():
    try:
        gpr=pagerank()
        #gpr=np.genfromtxt('GPR.txt',usecols=(1))
    except OSError:
        gpr=pagerank()
    return gpr

if __name__ == "__main__":
    print("Calculating pagerank")
    f=filewrite()
    print("done\n\n")
