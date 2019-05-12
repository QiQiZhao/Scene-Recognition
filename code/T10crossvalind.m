function accsum = T10crossvalind(data, datalabel,pre_method,class,lamda, ite,datalabel2)
[data_r, data_c] = size(data);
%generate the index of 10-fold
indices = crossvalind('Kfold',data_r,10); 
accsum =0;
%repeat 10 times to calculate the average accurancy
for i = 1 : 10
    %choose the learning method
    if pre_method ~= 'CNN'
        test = (indices == i);
        train = ~test;
        test_data = data(test, 1 : data_c);
        test_label = datalabel(test);
        train_data = data(train, 1 : data_c );
        train_label = datalabel(train);
    end
    if pre_method == 'CNN'
        test = (indices == i);
        train = ~test;
        [A,B,C,D]=size(data);
        indices = crossvalind('Kfold',D,10); 
        test_data = data(1:A, 1 : B,1:C,test);
        test_label = datalabel2(test);
        train_data = data(1:A, 1 : B,1:C,train);
        train_label = datalabel(train);
    end
    switch pre_method
        case 'naive_bayes'
             nbbb=fitcnb(train_data,train_label,'ClassNames',class,'DistributionNames','kernel');
             pretest=predict(nbbb,test_data);
             [a,~]=size(pretest);
             acc=0;
             for j=1:a
                 if(strcmp(pretest{j},test_label(j)))
                     acc=acc+1;
                 end
             end
             acc=acc/a;
        
        case 'NonlinearSVM'
            pretest=NLsvm_classify(train_data,train_label,test_data,class,lamda,ite);
            acc=0;
            [a,~]=size(pretest);
             for j=1:a
                 if(strcmp(pretest{j},test_label(j)))
                     acc=acc+1;
                 end
             end
             acc=acc/a;
             
        case 'knn'
            pretest=knn(train_data,train_label,test_data);
            acc=0;
            [a,~]=size(pretest);
             for j=1:a
                 if(strcmp(pretest{j},test_label(j)))
                     acc=acc+1;
                 end
             end
             acc=acc/a;
         
        case 'linearSVM'
            pretest=svm_classify(train_data,train_label,test_data,class,lamda,ite);
            acc=0;
            [a,~]=size(pretest);
             for j=1:a
                 if(strcmp(pretest{j},test_label(j)))
                     acc=acc+1;
                 end
             end
             acc=acc/a;
          
         case 'CNN'
            pretest=CNN(train_data,train_label,test_data);
            pretest=cellstr(pretest);
            acc=0;
            [a,~]=size(pretest);
             for j=1:a
                 if(strcmp(pretest{j},test_label(j)))
                     acc=acc+1;
                 end
             end
             acc=acc/a;
         
             
    end
    accsum=accsum+acc;
    
    
    
end
accsum=accsum/10;
end

