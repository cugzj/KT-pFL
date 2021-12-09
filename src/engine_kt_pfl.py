import os
import copy
import numpy as np
import torch
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
                    with_softmax = True,EarlyStopping=False,is_val = True,Temp=1.0):
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
                if iter == 0:
                    print('Temp:',Temp)
                output_logit = log_probs.float()/Temp
                ##
                #loss = SoftCrossEntropy(output_logit, labels)
                #loss = criterion(log_probs, labels)
                loss = KL_loss(output_logit, labels)
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
def get_models_logits(raw_logits, weight_alpha, N_models, penalty_ratio): #raw_logits为list-np；weight为tensor；
    weight_mean = torch.ones(N_models, N_models, requires_grad=True)
    weight_mean = weight_mean.float()/(N_models)
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    teacher_logits = torch.zeros(N_models, np.size(raw_logits[0],0), np.size(raw_logits[0],1), requires_grad=False) #创建logits of teacher  #next false
    models_logits = torch.zeros(N_models, np.size(raw_logits[0],0), np.size(raw_logits[0],1), requires_grad=True) #创建logits of teacher
    #weight.requires_grad = True #can not change requires_grad here
    weight = weight_alpha.clone()
    for self_idx in range(N_models): #对每个model计算其teacher的logits加权平均值
        teacher_logits_local = teacher_logits[self_idx]
        for teacher_idx in range(N_models): #对某一model，计算其他所有model的logits
            # if self_idx == teacher_idx:
            #     continue
            #teacher_tmp = weight[self_idx][teacher_idx] * raw_logits[teacher_idx]
            #teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * raw_logits[teacher_idx]) 
            #teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * torch.autograd.Variable(torch.from_numpy(raw_logits[teacher_idx]))) 
            #teacher_logits[self_idx] = torch.add(teacher_logits[self_idx], weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx])) 
            teacher_logits_local = torch.add(teacher_logits_local, weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx])) 
            #                                                                tensor中的一个像素点，本质标量 * teacher的完整logits
            
        loss_input = torch.from_numpy(raw_logits[self_idx])
        #loss_target = torch.autograd.Variable(teacher_logits[self_idx], requires_grad=True)   
        loss_target = teacher_logits_local                    
                                           
        loss = loss_fn(loss_input,loss_target)

        loss_penalty = loss_fn(weight[self_idx], weight_mean[self_idx])
        print('loss_penalty:', loss_penalty)     
        print('loss:', loss)
        loss += loss_penalty*penalty_ratio
        #loss = SoftCrossEntropy_without_logsoftmax(loss_input,loss_target)

        #weight[self_idx].zero_grad()
        #weight[self_idx].grad.zero_()
        weight.retain_grad() #保留叶子张量grad
        #print('weight.grad before loss.backward:', weight.grad)
        loss.backward(retain_graph=True)
        print('weight:', weight)
        #print('weight.requires_grad:', weight.requires_grad)
        #print('weight.grad:', weight.grad)
        #print('weight[self_idx]:', weight[self_idx])
        #print('weight[self_idx].grad:', weight[self_idx].grad)
        with torch.no_grad():
            #weight[self_idx] = weight[self_idx] - weight[self_idx].grad * 0.001  #更新权重
            gradabs = torch.abs(weight.grad)
            gradsum = torch.sum(gradabs)
            gradavg = gradsum.item() / (N_models)
            grad_lr = 1.0
            for i in range(5): #0.1
                if gradavg > 0.01:
                    gradavg = gradavg*1.0/5
                    grad_lr = grad_lr/5                
                if gradavg < 0.01:
                    gradavg = gradavg*1.0*5
                    grad_lr = grad_lr*5
            print('grad_lr:', grad_lr)
            weight.sub_(weight.grad*grad_lr)
            #weight.sub_(weight.grad*50)
            weight.grad.zero_()
    #############设定权重######################
    # set_weight_local = []
    # weight1 = [0.18, 0.18, 0.18, 0.18, 0.18, 0.02, 0.02, 0.02, 0.02, 0.02]
    # weight2 = [0.02, 0.02, 0.02, 0.02, 0.02, 0.18, 0.18, 0.18, 0.18, 0.18]
    # for i in range(N_models):
    #     if i <= 4:
    #         set_weight_local.append(weight1)
    #     if i >= 5:
    #         set_weight_local.append(weight2)
    # tensor_set_weight_local = torch.Tensor(set_weight_local)
    ###################################
    # 更新 raw_logits
    for self_idx in range(N_models): #对每个model计算其teacher的logits加权平均值
        weight_tmp = torch.zeros(N_models)
        idx_count = 0
        for teacher_idx in range(N_models): #对某一model，计算其softmax后的weight
            # if self_idx == teacher_idx:
            #     continue
            #weight加softmax#
            weight_tmp[idx_count] = weight[self_idx][teacher_idx]
            idx_count += 1
        #softmax_fn = nn.softmax() #这里不对，不应该softmax，应该normalization##先用低温softmax#
        weight_local = nn.functional.softmax(weight_tmp*5.0)

        idx_count = 0
        for teacher_idx in range(N_models): #对某一model，计算其他所有model的logits
            # if self_idx == teacher_idx:
            #     continue
            #models_logits[self_idx] = torch.add(models_logits[self_idx], weight[self_idx][teacher_idx] * torch.from_numpy(raw_logits[teacher_idx]))             
            #设定权重models_logits[self_idx] = torch.add(models_logits[self_idx], tensor_set_weight_local[self_idx][idx_count] * torch.from_numpy(raw_logits[teacher_idx]))
            models_logits[self_idx] = torch.add(models_logits[self_idx], weight_local[idx_count] * torch.from_numpy(raw_logits[teacher_idx]))            
            with torch.no_grad():
                #设定权重weight[self_idx][teacher_idx] = tensor_set_weight_local[self_idx][idx_count]                
                weight[self_idx][teacher_idx] = weight_local[idx_count]
            idx_count += 1             
    print('weight after softmax:', weight)
    #
    return models_logits, weight

#
def predict(model,dataarray,device,T):
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
            output_logit = Tsoftmax(logit.float()/T)

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


class KT_pFL():
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data,
                 KT_pFL_params,
                 model_init_params,
                 calculate_theoretical_upper_bounds_params,
                 device='cuda'):

        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = KT_pFL_params['N_alignment']

        self.N_rounds = KT_pFL_params['N_rounds']
        self.N_logits_matching_round = KT_pFL_params['N_logits_matching_round']
        self.logits_matching_batchsize = KT_pFL_params['logits_matching_batchsize']
        self.N_private_training_round = KT_pFL_params['N_private_training_round']
        self.private_training_batchsize = KT_pFL_params['private_training_batchsize']
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
        self.Temp = model_init_params['Temp']
        self.penalty_ratio = model_init_params['penalty_ratio']

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
        #weight_alpha = weight_alpha.float()/(self.N_parties-1)        
        weight_alpha = weight_alpha.float()/(self.N_parties)        
        #
        newdata = read_data("fmnist")
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
                logits.append(predict(model, X_data, device,self.Temp))
            #logits = np.sum(logits,axis =0)
            #logits /= self.N_parties
            #
            print('before get logits:', weight_alpha)
            logits_models, weight_alpha = get_models_logits(logits, weight_alpha, self.N_parties, self.penalty_ratio)
            #print('after get logits:', weight_alpha)
            # test performance
            logits_models = logits_models.detach().numpy() 

            print("test performance ... ")

            for index, model in enumerate(self.collaborative_parties):
                X_train, y_train, X_test, y_test = read_user_data(index, newdata, "Mnist")
                dataloader =  DataLoader(data_new(X_test,y_test), batch_size=8,
                                          shuffle=True,
                                          sampler=None, batch_sampler=None, num_workers=self.num_workers, drop_last=False)
                acc = val_one_model(model, dataloader, criterion=None, device=torch.device('cuda'))

                collaboration_performance[index].append(acc)
                print(collaboration_performance[index][-1])

                with open('./result/fmnist_ours_with_20models_4labels_{}local_{}dis_T{}_p{}_N{}.txt'.format(self.N_private_training_round,self.N_logits_matching_round,int(self.Temp),self.penalty_ratio,self.N_alignment), 'a') as f:
                    f.write('{}\t{}\t{}\n'.format(r, index, acc))

            r += 1
            if r > (self.N_rounds*2):
                break

            print("updates models ...")
            #

            if r == 1:
                local_epoch= 1 - 1
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
                                    with_softmax = False,EarlyStopping=False, is_val=False, Temp=self.Temp)

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
