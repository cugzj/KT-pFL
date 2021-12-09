import os
import copy
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import SGD,Adam
from data_utils import generate_alignment_data,data,data_new
from model_utils import read_data, read_user_data

def KL_loss(inputs, target, reduction='average'):
    log_likelihood = F.log_softmax(inputs, dim=1)
    #print('log_probs:',log_likelihood)
    #batch = inputs.shape[0]
    if reduction == 'average':
        #loss = torch.sum(torch.mul(log_likelihood, target)) / batch
        loss = F.kl_div(log_likelihood, target, reduction='mean')
    else:
        #loss = torch.sum(torch.mul(log_likelihood, target))
        loss = F.kl_div(log_likelihood, target, reduction='sum')
    return loss

def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

def train_one_model(model,train_dataloader,test_dataloader,optimizer,epoch,device,criterion, min_delta=0.01,patience=3,
                    with_softmax = True,EarlyStopping=False,is_val = True):
    model.to(device)
    all_train_loss, all_train_acc, all_val_loss, all_val_acc = [],[],[],[]
    for iter in range(epoch):
        model.train()

        train_loss = []
        train_acc = []
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            #print('images.shape:',images.shape)
            log_probs = model(images)
            if with_softmax:
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                acc = torch.mean((labels ==torch.argmax(log_probs,dim=-1)).to(torch.float32))
                train_acc.append(acc.item())
                train_loss.append(loss.item())
            else:
                #Tsoftmax = nn.Softmax(dim=1)
                #加入温度系数T#
                output_logit = log_probs.float()/1.0
                ##
                loss = KL_loss(output_logit, labels)
                #loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
        if with_softmax:
            all_train_loss.append(sum(train_loss)/len(train_loss))
            all_train_acc.append(sum(train_acc)/len(train_acc))
            if is_val:
                val_loss,val_acc = val_one_model(model,test_dataloader,criterion,device)
                print('val_acc:',val_acc, 'val_loss:', val_loss)
                all_val_loss.append(val_loss)
                all_val_acc.append(val_acc)
                #if EarlyStopping and len(all_val_acc)>patience:
                    #if max(all_val_acc[-patience:])-min(all_val_acc[-patience:])<=min_delta:
                        #break
    return all_train_loss,all_train_acc,all_val_loss,all_val_acc

def val_one_model(model,dataloader,criterion=None,device= torch.device('cuda')):
    model.eval()
    acc = []
    loss_out = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            log_probs = model(images)
            if criterion is not None:
                loss = criterion(log_probs, labels)
                loss_out.append(loss.item())
            #print('log_probs.shape:',log_probs.shape)
            acc_ = torch.mean((labels == torch.argmax(log_probs, dim=-1)).to(torch.float32))
            acc.append(acc_.item())
        if criterion is not None:
            return sum(loss_out)/len(loss_out),sum(acc)/len(acc)
        else:
            return sum(acc)/len(acc)
#
def partitionOfK(numbers, start, end, k):
  if k < 0 or numbers == [] or start < 0 or end >= len(numbers) or k > end:
    return None
  low = start
  high = end
  key = numbers[start]
  while low < high:
    while low < high and numbers[high] >= key:
      high -= 1
    numbers[low] = numbers[high]
    while low < high and numbers[low] <= key:
      low += 1
    numbers[high] = numbers[low]
  numbers[low] = key
  if low < k:
    return partitionOfK(numbers, start + 1, end, k)
  elif low > k:
    return partitionOfK(numbers, start, end - 1, k)
  else:
    return numbers[low]

def get_cosdist(a,b):
    if len(a) != len(b):
        return None
    part_up = 0.0
    a_sq = 0.0
    b_sq = 0.0
    for a1, b1 in zip(a,b):
        part_up += a1*b1
        a_sq += a1**2
        b_sq += b1**2
    part_down = math.sqrt(a_sq*b_sq)
    if part_down == 0.0:
        return None
    else:
        return part_up / part_down
    return 

def get_models_logits(raw_logits, threshold, N_models):  #input:  

    flatten_logits = []
    cosdist = []
    tmp_cosdist = []
    tmp_logits = []
    models_logits = []
    # threshold = -1 ############# CHANGE ###############
    add_model_count = 0
    range_models = N_models

    for index in range(N_models):
        flatten_logits.append(raw_logits[index].flatten())

    for index in range(N_models):
        for index2 in range(range_models):
            if index == index2:
                tmp_cosdist.append(1)
            else:
                tmp_cosdist.append(get_cosdist(flatten_logits[index],flatten_logits[index2]))
        cosdist.append(tmp_cosdist)   #cosdist = [model1's all cosdist list,model2's all cosdist list,...]
        tmp_cosdist = []

    print("cosdist:")
    print(cosdist)
    thresholdtopk=[]
    deepcopy_cosdist = copy.deepcopy(cosdist)
    for index in range(N_models):
        thresholdtopk.append(partitionOfK(deepcopy_cosdist[index], 0, len(cosdist[index]) - 1, int(len(cosdist[index])/2)-1))
    print("thresholdtopk:", thresholdtopk)
    #judge threshold
    
    

    for index in range(N_models):
        for index2 in range(range_models):
            #if cosdist[index][index2] > threshold: #cosdist (-1,1), cosdist small means angle big
            if cosdist[index][index2] > thresholdtopk[index]:
                print("index,index2,cosdist[index][index2],thresholdtopk[index]:",index,index2,cosdist[index][index2],thresholdtopk[index])
                tmp_logits.append(raw_logits[index2])
                print("model_id_of_model{0}_used:".format(index), index2)
                add_model_count += 1
        print("model_num of model {0} used".format(index))
        print("-- {0} --".format(add_model_count))
        tmp_logits = np.sum(tmp_logits,axis=0)
        tmp_logits /= add_model_count
        models_logits.append(tmp_logits)
        add_model_count = 0
        tmp_logits = []

    return models_logits
#
def predict(model,dataarray,device):
    model.eval()
    out= []
    bs = 32
    dataarray = dataarray.astype(np.float32)
    with torch.no_grad():
        for ind in range(0,len(dataarray),bs):
            data = dataarray[ind:(ind+bs)]
            data = torch.from_numpy(data).to(device)

            logit = model(data)

            Tsoftmax = nn.Softmax(dim=1)
            #加入温度系数T#
            output_logit = Tsoftmax(logit.float()/1.0)

            out.append(output_logit.cpu().numpy())
    out = np.concatenate(out)
    return out

def train_models(models, X_train, y_train, X_test, y_test,
                 device = 'cpu',save_dir = "./", save_names = None,
                 early_stopping = True, min_delta = 0.001,num_workers=0,
                 batch_size = 128, epochs = 20, is_shuffle=True,patience=3
                ):
    '''
    Train an array of models on the same dataset. 
    We use early termination to speed up training. 
    '''
    
    resulting_val_acc = []
    record_result = []

    for n, model in enumerate(models):
        print("Training model ", n)
        model.to(device)
        train_dataloader = DataLoader(data(X_train, y_train), batch_size=batch_size,shuffle=is_shuffle,
                                      sampler=None,batch_sampler= None,num_workers= num_workers,drop_last = False)
        test_dataloader = DataLoader(data(X_test, y_test), batch_size=batch_size,
                                      sampler=None,batch_sampler= None,num_workers= num_workers,drop_last = False)
        optimizer = SGD(model.parameters(),lr=0.02)
        criterion = nn.CrossEntropyLoss().to(device)
        train_loss,train_acc,val_loss,val_acc = train_one_model(model, train_dataloader,test_dataloader, optimizer,
                                                                epochs, device, criterion,  min_delta,patience,
                        EarlyStopping=early_stopping,is_val=True)
        
        resulting_val_acc.append(val_acc[-1])
        record_result.append({"train_acc": train_acc,
                              "val_acc": val_acc,
                              "train_loss": train_loss,
                              "val_loss": val_loss})

        if save_dir is not None:
            save_dir_path = os.path.abspath(save_dir)
            #make dir
            os.makedirs(save_dir_path,exist_ok=True)
            if save_names is None:
                file_name = os.path.join(save_dir_path , "model_{0}".format(n) + ".pt")
            else:
                file_name = os.path.join(save_dir_path , save_names[n] + ".pt")
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epochs}
            torch.save(state,file_name)

    
    print("pre-train accuracy: ")
    print(resulting_val_acc)
        
    return record_result


class KT_pFL_topk():
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data,
                 KT_pFL_topk_params,
                 model_init_params,
                 calculate_theoretical_upper_bounds_params,
                 device='cuda'):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = KT_pFL_topk_params['N_alignment']

        self.N_rounds = KT_pFL_topk_params['N_rounds']
        self.N_logits_matching_round = KT_pFL_topk_params['N_logits_matching_round']
        self.logits_matching_batchsize = KT_pFL_topk_params['logits_matching_batchsize']
        self.N_private_training_round = KT_pFL_topk_params['N_private_training_round']
        self.private_training_batchsize = KT_pFL_topk_params['private_training_batchsize']
        self.device = device

        print("calculate the theoretical upper bounds for participants: ")
        self.upper_bounds = []
        self.pooled_train_result = []
        #
        # 参数
        epochs = calculate_theoretical_upper_bounds_params['epochs']
        min_delta= calculate_theoretical_upper_bounds_params['min_delta']
        patience= calculate_theoretical_upper_bounds_params['patience']
        batch_size= calculate_theoretical_upper_bounds_params['batch_size']
        is_shuffle= calculate_theoretical_upper_bounds_params['is_shuffle']
        num_workers = calculate_theoretical_upper_bounds_params['num_workers']
        '''
        for model in parties:
            model_ub = copy.deepcopy(model)
            print('total_private_data["X"]:', total_private_data["X"].shape)
            train_dataloader = DataLoader(data(total_private_data["X"], total_private_data["y"]), batch_size=batch_size,
                                          shuffle=is_shuffle,
                                          sampler=None, batch_sampler=None, num_workers=num_workers, drop_last=False)
            test_dataloader = DataLoader(data(private_test_data["X"], private_test_data["y"]), batch_size=batch_size,
                                         sampler=None, batch_sampler=None, num_workers=num_workers, drop_last=False)

            optimizer = SGD(model_ub.parameters(), lr=0.005)
            criterion = nn.CrossEntropyLoss()


            train_loss, train_acc, val_loss, val_acc = train_one_model(model_ub, train_dataloader, test_dataloader,
                                                                       optimizer,
                                                                       epochs, self.device, criterion, min_delta, patience,
                                                                       EarlyStopping=False, is_val=True)


            self.upper_bounds.append(val_acc[-1])
            self.pooled_train_result.append({"val_acc": val_acc,
                                             "acc": train_acc})

            del model_ub
        print("the upper bounds are:", self.upper_bounds)
        '''
        self.collaborative_parties = []
        self.init_result = []

        print("start model initialization: ")

        epochs = model_init_params['epochs']
        #epochs = 1
        min_delta= model_init_params['min_delta']
        patience= model_init_params['patience']
        batch_size= model_init_params['batch_size']
        is_shuffle= model_init_params['is_shuffle']
        num_workers = model_init_params['num_workers']
        self.num_workers =num_workers

        newdata = read_data("fmnist") 
        
        for i in range(self.N_parties):
            X_train, y_train, X_test, y_test = read_user_data(i, newdata, "Mnist")
            model = parties[i]
            
            train_dataloader = DataLoader(data_new(X_train,y_train), batch_size=batch_size, shuffle=is_shuffle,
                                          sampler=None, batch_sampler=None, num_workers=num_workers, drop_last=False)
            test_dataloader = DataLoader(data_new(X_test,y_test), batch_size=batch_size,
                                         sampler=None, batch_sampler=None, num_workers=num_workers, drop_last=False)
            optimizer = SGD(model.parameters(), lr=0.02)
            criterion = nn.CrossEntropyLoss()


            train_loss, train_acc, val_loss, val_acc = train_one_model(model, train_dataloader, test_dataloader,
                                                                       optimizer,
                                                                       epochs, self.device, criterion, min_delta, patience,
                                                                       EarlyStopping=False, is_val=True)

            self.collaborative_parties.append(model)

            self.init_result.append({"val_acc": val_acc,
                                     "train_acc": train_acc,
                                     "val_loss": val_loss,
                                     "train_loss": train_loss,
                                     })
            print('val_acc:',i,val_acc)
        # print('model initialization are:')
        # END FOR LOOP
        
        print("finish model initialization: ")
    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        r = 0
        device = torch.device('cuda')
        #
        weight_alpha = torch.ones(self.N_parties, self.N_parties, requires_grad=True)
        weight_alpha = weight_alpha.float()/(self.N_parties-1)        
        #
        newdata = read_data("fmnist")
        threshold = -1
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)

            print("round ", r)

            print("update logits ... ")
            # update logits
            logits = []
            for model in self.collaborative_parties:
                X_data = copy.deepcopy(alignment_data["X"])
                if len(X_data.shape)==4:
                    X_data = np.transpose(X_data, (0, 3, 1, 2))
                else:
                    X_data = np.repeat(X_data[:,None],repeats=3,axis=1)
                logits.append(predict(model, X_data, device))
            #logits = np.sum(logits,axis =0)
            #logits /= self.N_parties
            #
            #print('before get logits:', weight_alpha)
            #logits_models, weight_alpha = get_models_logits(logits, weight_alpha, self.N_parties)
            logits_models = get_models_logits(logits,threshold,self.N_parties)
            #print('after get logits:', weight_alpha)
            # test performance
            #logits_models = logits_models.detach().numpy() 

            print("test performance ... ")

            for index, model in enumerate(self.collaborative_parties):
                X_train, y_train, X_test, y_test = read_user_data(index, newdata, "Mnist")
                dataloader =  DataLoader(data_new(X_test,y_test), batch_size=8,
                                          shuffle=True,
                                          sampler=None, batch_sampler=None, num_workers=self.num_workers, drop_last=False)
                acc = val_one_model(model, dataloader, criterion=None, device=torch.device('cuda'))

                collaboration_performance[index].append(acc)
                print(collaboration_performance[index][-1])

                with open('./result/fmnist_topk_20models_4labels_15local.txt', 'a') as f:
                    f.write('{}\t{}\t{}\n'.format(r, index, acc))

            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            #

            if r == 1:
                local_epoch= 4 - 1
            if local_epoch < self.N_private_training_round:
                local_epoch += 1
            else:
                local_epoch = self.N_private_training_round

            for index, model in enumerate(self.collaborative_parties):
                print("model {0} starting alignment with public logits... ".format(index))
                X_train, y_train, X_test, y_test = read_user_data(index, newdata, "Mnist")

                train_dataloader = DataLoader(data(alignment_data["X"], logits_models[index]), batch_size=self.logits_matching_batchsize,
                                              shuffle=True,
                                              sampler=None, batch_sampler=None, num_workers=self.num_workers, drop_last=False)
                test_dataloader = None
                optimizer = SGD(model.parameters(), lr=0.02)
                criterion = nn.MSELoss()
                epoch = self.N_logits_matching_round


                train_one_model(model, train_dataloader, test_dataloader, optimizer, epoch, self.device, criterion,
                                    with_softmax = False,EarlyStopping=False, is_val=False)

                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))

                train_dataloader = DataLoader(data_new(X_train,y_train),
                                              batch_size=self.private_training_batchsize,shuffle=True,
                                              sampler=None, batch_sampler=None, num_workers=self.num_workers, drop_last=False)
                test_dataloader = None
                optimizer = SGD(model.parameters(), lr=0.02)
                criterion = nn.CrossEntropyLoss()
                epoch = self.N_private_training_round

                train_one_model(model, train_dataloader, test_dataloader, optimizer, local_epoch, self.device, criterion,
                                    EarlyStopping=False, is_val=False)

                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance
