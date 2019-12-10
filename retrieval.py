import os
import pagerank
import qtspr
import ptspr
import time
import numpy as np

#make a dictionary to get whole information from indri-list
def get_info():
    path = "data/indri-lists/"
    file_list = os.listdir(path)
    file_list_txt = [file for file in file_list if file.endswith(".txt")]
    info={}
    score=[]
    doc_id=[]
    doc_for_index=[]
    for txt_name in file_list_txt:
        txt = open('data/indri-lists/'+txt_name, 'r')
        query_temp=txt_name.split('.')[0].split('-')
        query=txt_name.split('.')[0]
        sorted_id=int(query_temp[0]+query_temp[1])
       # print(query_id_temp)
        doc_id_l=[]
        rank_l=[]
        score_l=[]
        #merge data into modified query_id
        #In benchmark program.. query_id sorting was not effective..
        info[sorted_id]={}
        info[sorted_id]['query_id']=query
        info[sorted_id]["doc_id"]=list()
        info[sorted_id]["rank"]=list()
        info[sorted_id]["score"]=dict()
        for i in txt:
            temp = i.split(' ')
            #a=[]
            #doc_id=int(temp[2])-1
            #a.append(int(temp[2])-1)
            #a.append(temp[4])
            info[sorted_id]["doc_id"].append(int(temp[2])-1)
            info[sorted_id]["rank"].append(int(temp[3]))
            info[sorted_id]["score"][int(temp[2])-1]=temp[4]
            score.append(float(temp[4]))
            doc_id.append(int(temp[2])-1)
            doc_for_index.append(int(temp[2]))
    return info,score,doc_id,doc_for_index

    #NS_GPR
def gpr_ns():
    import time
    pr = pagerank.loadgpr()
    start_ns=time.time()
    info,_,_,_=get_info()
    f = open('GPR-NS.txt', 'w')
    for i in sorted(info):
        query_id = info[i]['query_id']
        doc_id = info[i]["doc_id"]
        pr_score = np.argsort(pr[doc_id])[::-1].tolist() #sorting score
        doc_id_ = np.array(doc_id) # make doc array
        pr_rank = doc_id_[pr_score] #get rank by index of score
        rank =1
        for j in pr_rank:
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, j+1, rank, pr[j]))
            rank += 1
    f.close()
    print("{} secs for GPR-NS".format(round(time.time() - start_ns,4)),"/ round(time,4)")
    print("GPR-NS is done \n")

    
def gpr_ws():
    #WS_GPR
    import time
    pr = pagerank.loadgpr()
    start_ws=time.time()
    info,_,_,_=get_info()
    f = open('GPR-WS.txt', 'w')
    for i in sorted(info):
        query_id = info[i]['query_id']
        doc_id = info[i]["doc_id"]
        list_sum=[]
        a=0.8
        for _,j in enumerate(doc_id):
      #  doc_id = info[i]["doc_id"]
            score_=float(info[i]["score"][j])
        #score=np.array(score_, dtype=np.float64)
            sum_=np.multiply(a,pr[j]).sum(axis=0)+np.multiply((1-a),score_).sum(axis=0)
            list_sum.append(sum_)
        pr_score = np.argsort(list_sum)[::-1].tolist()
        doc_id_ = np.array(doc_id) # make doc array
        pr_rank = doc_id_[pr_score] #get rank by index of score
        rank =1
        for k in pr_rank:
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, k+1, rank, list_sum[doc_id.index(k)]))
            rank += 1
    f.close()
    print("{} secs for GPR-WS".format(round(time.time() - start_ws,4)),"/ round(time,4)")
    print("GPR-WS is done \n")

# get_path, get_id, get_score => made for cm method
def get_path():
    data_path = "data/indri-lists/"
    data_files =  os.listdir(data_path)
    dic = {}
    for i in data_files:
        query_id = i.split('.')[0]
        sorted_id = int(query_id.split('-')[0] + query_id.split('-')[1])
        dic[sorted_id] = [query_id, data_path+i]
    # dictionary with query_id and data_path to call
    return dic

def get_id(path):
    file = open(path, 'r')
    doc = []
    for i in file:
        doc.append(int(i.split(' ')[2]) - 1)
    return doc #doc_id from path

def get_score(path):
    file = open(path, 'r')
    score = []
    for i in file:
        score.append(float(i.split(' ')[4]))
    return score  #score from path

def gpr_cm():
    #custom_pagerank
    import time
    pr = pagerank.loadgpr()
    start_cm=time.time()
    path = get_path()
    _,score,_,_=get_info()
    min_s,max_s=np.min(score),np.max(score)

    f = open('GPR-CM.txt', 'w')
    for cur_num in sorted(path):
        query_id = path[cur_num][0]
        file_name = path[cur_num][1]
        doc_id=get_id(file_name)
        ir_score=get_score(file_name)
        ir_score=[(float(i)-min_s)/(max_s-min_s) for i in ir_score] # min-max scaling ; ir score
        gpr=pr[doc_id]*(-1)
        gpr=np.sort(gpr) # sorting for multiply weight
        gpr=[(float(i)+np.max(pr))/(-np.max(pr)+np.min(pr)) for i in gpr] # min-max scaling to minus of pagerank value
        #gpr=(-1)*gpr
        
        ## for modified weight
        alpha=0.1 # start point of modified weight numpy
        beta=0.5 # peak point of modified weight numpy
        gamma=0.2 # relative location of peak point at modified weight
        omega=0.3 # end point of modified weight numpy
        l=len(doc_id)
        multi1=np.arange(alpha,beta,(beta-alpha)/((1-omega)*l))
        multi2=np.arange(omega,beta,(beta-omega)/(omega*l))[::-1]
        multi=np.hstack((multi1,multi2))
        # min-max scaling but add 0.01 to avoid zero weight
        multi=[(float(i)-np.min(multi)+0.01)/(np.max(multi)-np.min(multi)) for i in multi]
        if len(multi) >l:
            multi=multi[:l]
        #cm_pr=np.multiply(multi,gpr)+ir_score
        cm_pr=np.multiply(multi,gpr)+ir_score
        gpr_score = np.argsort(cm_pr)[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        gpr_rank = doc_id_arr[gpr_score]
        rank_num = 0
        for i in gpr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, i + 1, rank_num, cm_pr[doc_id.index(i)]))
    f.close()
    print("{} secs for GPR-CM".format(round(time.time() - start_cm,4)),"/ round(time,4)")
    print("GPR-CM is done \n")
    
def ptspr_ns():
    #NS_PTSPR
    import time
    pr = ptspr.ptspr_on()
    start=time.time()
    info,_,_,_=get_info()
    f = open('PTSPR-NS.txt', 'w')
    query=0 # add query for indexing
    for i in sorted(info):
        query_id = info[i]['query_id']
        doc_id = info[i]["doc_id"]
        pr_score = np.argsort(pr[query][doc_id])[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        pr_rank = doc_id_arr[pr_score]

        rank =1
        for j in pr_rank:
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, j+1, rank, pr[query][j]))
            rank += 1
        query+=1
    f.close()
    print("{} secs for PTSPR-NS".format(round(time.time() - start,4)),"/ round(time,4)")
    print("PTSPR-NS is done \n")

    
def ptspr_ws():
    #WS_ptspr
    import time
    pr = ptspr.ptspr_on()
    start=time.time()
    info,_,_,_=get_info()
    f = open('PTSPR-WS.txt', 'w')
    query=0 # add query for indexing
    for i in sorted(info):
        query_id = info[i]['query_id']
        doc_id = info[i]["doc_id"]
        list_sum=[]
        a=0.8
        for _,j in enumerate(doc_id):
      #  doc_id = info[i]["doc_id"]
            score_=float(info[i]["score"][j])
        #score=np.array(score_, dtype=np.float64)
            sum_=np.multiply(a,pr[query][j]).sum(axis=0)+np.multiply((1-a),score_).sum(axis=0)
            list_sum.append(sum_)
        pr_score = np.argsort(list_sum)[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        pr_rank = doc_id_arr[pr_score]
        rank =1
        for k in pr_rank:
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, k+1, rank, list_sum[doc_id.index(k)]))
            rank += 1
        query+=1
    f.close()
    print("{} secs for PTSPR-WS".format(round(time.time() - start,4)),"/ round(time,4)")
    print("PTSPR-WS is done \n")
    
    
    
def ptspr_cm():
    #custom_ptspr
    import time
    pr = ptspr.ptspr_on()
    start=time.time()
    path = get_path()
    _,score,_,_=get_info()
    min_s,max_s=np.min(score),np.max(score)
    query=0
    f = open('PTSPR-CM.txt', 'w')
    for cur_num in sorted(path):
        query_id = path[cur_num][0]
        file_name = path[cur_num][1]
        # doc id in the current indri file
        doc_id=get_id(file_name)
        ir_score=get_score(file_name)
        ir_score=[(float(i)-min_s)/(max_s-min_s) for i in ir_score] # min-max scaling ; ir score
        gpr=pr[query][doc_id]*(-1)
        gpr=np.sort(gpr)
        gpr=[(float(i)+np.max(pr))/(-np.max(pr)+np.min(pr)) for i in gpr] # min-max scaling to minus of pagerank value
        #gpr=(-1)*gpr
        alpha=0.1
        beta=0.5
        gamma=0.2
        omega=0.3
        l=len(doc_id)
        multi1=np.arange(alpha,beta,(beta-alpha)/((1-omega)*l))
        multi2=np.arange(omega,beta,(beta-omega)/(omega*l))[::-1]
        multi=np.hstack((multi1,multi2))
        # min-max scaling but add 0.01 to avoid zero weight
        multi=[(float(i)-np.min(multi)+0.01)/(np.max(multi)-np.min(multi)) for i in multi]
        if len(multi) >l:
            multi=multi[:l]
        cm_pr=np.multiply(multi,gpr)+ir_score
        # sort by descending order
        gpr_score = np.argsort(cm_pr)[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        gpr_rank = doc_id_arr[gpr_score]
        rank_num = 0
        for i in gpr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, i + 1, rank_num, cm_pr[doc_id.index(i)]))
        query+=1
    f.close()
    print("{} secs for PTSPR-CM".format(round(time.time() - start,4)),"/ round(time,4)")
    print("PTSPR-CM is done \n")

def qtspr_ns():
    #NS_QTSPR
    import time
    pr = qtspr.qtspr_on()
    start=time.time()
    info,_,_,_=get_info()
    f = open('QTSPR-NS', 'w')
    query=0
    for i in sorted(info):
        query_id = info[i]['query_id']
        doc_id = info[i]["doc_id"]
        pr_score = np.argsort(pr[query][doc_id])[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        pr_rank = doc_id_arr[pr_score]

        rank =1
        for j in pr_rank:
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, j+1, rank, pr[query][j]))
            rank += 1
        query+=1
    f.close()
    print("{} secs for QTSPR-NS".format(round(time.time() - start,4)),"/ round(time,4)")
    print("QTSPR-NS is done \n")

    
def qtspr_ws():
    #WS_qtspr
    import time
    pr = qtspr.qtspr_on()
    start=time.time()
    info,_,_,_=get_info()
    f = open('QTSPR-WS.txt', 'w')
    query=0
    for i in sorted(info):
        query_id = info[i]['query_id']
        doc_id = info[i]["doc_id"]
        list_sum=[]
        a=0.8
        for _,j in enumerate(doc_id):
      #  doc_id = info[i]["doc_id"]
            score_=float(info[i]["score"][j])
        #score=np.array(score_, dtype=np.float64)
            sum_=np.multiply(a,pr[query][j]).sum(axis=0)+np.multiply((1-a),score_).sum(axis=0)
            list_sum.append(sum_)
        pr_score = np.argsort(list_sum)[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        pr_rank = doc_id_arr[pr_score]
        rank =1
        for k in pr_rank:
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, k+1, rank, list_sum[doc_id.index(k)]))
            rank += 1
        query+=1
    f.close()
    print("{} secs for QTSPR-WS".format(round(time.time() - start,4)),"/ round(time,4)")
    print("QTSPR-WS is done \n")

    
def qtspr_cm():
    #custom_qtspr
    import time
    pr = qtspr.qtspr_on()
    start=time.time()
    path = get_path()
    _,score,_,_=get_info()
    min_s,max_s=np.min(score),np.max(score)
    query=0
    f = open('QTSPR-CM.txt', 'w')
    for cur_num in sorted(path):
        query_id = path[cur_num][0]
        file_name = path[cur_num][1]
        # doc id in the current indri file
        doc_id=get_id(file_name)
        ir_score=get_score(file_name)
        ir_score=[(float(i)-min_s)/(max_s-min_s) for i in ir_score]
        gpr=pr[query][doc_id]*(-1)
        gpr=np.sort(gpr)
        gpr=[(float(i)+np.max(pr))/(-np.max(pr)+np.min(pr)) for i in gpr]
        #gpr=(-1)*gpr
        alpha=0.1
        beta=0.5
        gamma=0.2
        omega=0.3
        l=len(doc_id)
        multi1=np.arange(alpha,beta,(beta-alpha)/((1-omega)*l))
        multi2=np.arange(omega,beta,(beta-omega)/(omega*l))[::-1]
        multi=np.hstack((multi1,multi2))
                # min-max scaling but add 0.01 to avoid zero weight
        multi=[(float(i)-np.min(multi)+0.01)/(np.max(multi)-np.min(multi)) for i in multi]
        if len(multi) >l:
            multi=multi[:l]
        cm_pr=np.multiply(multi,gpr)+ir_score
        gpr_score = np.argsort(cm_pr)[::-1].tolist()
        doc_id_arr = np.array(doc_id)
        gpr_rank = doc_id_arr[gpr_score]
        rank_num = 0
        for i in gpr_rank:
            rank_num += 1
            f.write("{} Q0 {} {} {} run-1\n".format(query_id, i + 1, rank_num, cm_pr[doc_id.index(i)]))
        query+=1
    f.close()
    print("{} secs for QTSPR-CM".format(round(time.time() - start,4)),"/ round(time,4)")
    print("QTSPR-CM is done \n")
    
if __name__ == "__main__":
    print("Calculating retrievals.....\n")
    pagerank.filewrite()
    qtspr.filewrite()
    ptspr.filewrite()
    print("\n")
    gpr_ns()
    gpr_ws()
    gpr_cm()
    ptspr_ns()
    ptspr_ws()
    ptspr_cm()
    qtspr_ns()
    qtspr_ws()
    qtspr_cm()
    print("Done!")
