 % CS 534 - Machine Learning 
 % Final Project Part_3
 % Chenyu Wang ( ID: 932-079-604 )
 % Hongyan Yi (ID: 932-430-243 )
 % Qun Jing (ID:932-011-106 )
  
 clc
 clear
 
 
 load('devmtx.mat');
 load('trnmtx.mat');
 load('tstmtx.mat');
 load('Prune_Index.mat');
 
 devmtx = de_d.tfidf;
 devmtx_p = devmtx(P_Vocidx',:);
 devcat = de_d.cat;
 trnmtx = tr_d.tfidf;
 trnmtx_p = trnmtx(P_Vocidx',:);
 trncat = tr_d.cat;
 tstmtx = ts_d.tfidf;
 tstmtx_p = tstmtx(P_Vocidx',:);
 tstcat = ts_d.cat;
  
for k=1:50                              
    for n=1:size(devmtx,2)                % loops over developing group
        % computes cosine similarities between a dev sample and all train
        cosine_s = devmtx(:,n)'*trnmtx;
        % sorts the resulting cosine similarities in descending order
        [void,order] = sort(cosine_s,'descend');
        % gets the categories for the k nearest neighbors
        vals = trncat(order(1:k));
        % constructs a histogram for the k nearest neighbor categories
        hcat = hist(vals,1:length(unique(trncat)));
        % gets the most common category among the k nearest neighbors
        [void,thecat] = max(hcat);
        % assigns the corresponding dev sample to the most common category
        assignedcat(n,1) = thecat;
    end;
    % computes the accuracy for the current value of k
	accuracy(k) = sum(devcat==assignedcat)/length(devcat)*100;
end;
   
[maxaccuracy,koptim] = max(accuracy);
fprintf('When k = %3.1f, our max accuracy will reach %4.2f\n\n', koptim, maxaccuracy)
   
for  n=1:size(tstmtx,2) % applies knn (with koptim) to the test set
    [void,order_t] = sort(tstmtx(:,n)'*trnmtx,'descend');
    [void,order_p] = sort(tstmtx_p(:,n)'*trnmtx_p,'descend');
    vals = trncat(order_t(1:koptim));
    vals_p = trncat(order_p(1:koptim));
    hcat = hist(vals,1:length(unique(trncat)));
    hcat_p = hist(vals_p,1:length(unique(trncat)));
    [void,thecat_t] = max(hcat);
    [void,thecat_p] = max(hcat_p);
    assignedcat_t(n,1) = thecat_t;
    assignedcat_p(n,1) = thecat_p;
end;

confusion_mtx = zeros(3,3);
for k=1:3 
    for n=1:3
        confusion_mtx(k,n) = sum((assignedcat_t==k)&(tstcat==n));
        confusion_mtx_p(k,n) = sum((assignedcat_p==k)&(tstcat==n));
    end;
end;
 
string = '                   Cat1 Cat2 Cat3';
CM = 0; 
for k=1:3
    formatted_txt = '%s\nAssigned to Cat %d:  %2d  %2d  %2d';
    string = sprintf(formatted_txt,string,k,confusion_mtx(k,:));
    CM = CM +confusion_mtx(k,k);
end;
    disp(string);
    
    Acc = CM/1.2;
    
    fprintf('The accuracy of the testing data is %4.2f.\n\n',Acc); 

string = '                                 Cat1 Cat2 Cat3';
CM_p = 0; 
for k=1:3
    formatted_txt = '%s\nAfter pruning, assigned to Cat %d:  %2d  %2d  %2d';
    string = sprintf(formatted_txt,string,k,confusion_mtx_p(k,:));
    CM_p = CM_p +confusion_mtx_p(k,k);
end;
    disp(string);
    
    Acc_p = CM_p/1.2;
    
    fprintf('After pruning, the accuracy of the testing data is %4.2f.\n',Acc_p); 
