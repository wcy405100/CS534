% CS 534 - Machine Learning 
% Final Project Part_2
% Chenyu Wang ( ID: 932-079-604 )
% Hongyan Yi (ID: 932-430-243 )
% Qun Jing (ID:932-011-106 )


 clc
 clear
 
 load('voc.mat');
 load('data1.mat');
 load('data2.mat');
 load('data3.mat');
 
 vocabulary_total = [data1.vocab,data2.vocab,data3.vocab];
 [vocabulary_t,void,index_v] = unique(vocabulary_total);
 vocabulary_size=length(vocabulary_t);
 frequencies_total = hist(index_v,vocabulary_size);
 P_Vocidx = (frequencies_total)>2;                          % make a threshold for words appearance>2
 Stop_w = {'a' 'an' 'the' 'they' 'i' 'you' 'where' 'when' 'what' 'how' 'is' 'are' 'be'...
     'of' 'to' 'that' 'not' };                               % build a dictoinary for stop words
 
 for k = 1:length(Stop_w)
     S_Vocidx(1,k) = find(strcmp(vocabulary_t,Stop_w(k)));
 end
 P_Vocidx(S_Vocidx) = 0;                                    % get a logical array with pruned vocab = 0 and remaining = 1
 vocabulary_p = vocabulary_t(P_Vocidx);                     % pruned vocabulary cell array 

for m = 1:9                                                 % make 9 iteration for a robust result
 
 ridx1 = randperm(length(data1));
 ridx2 = randperm(length(data2));
 ridx3 = randperm(length(data3));
 
 randomizetst = randperm(120);
 randomizedev = randperm(120);
 randomizetrn = randperm(length(data1)+length(data2)+length(data3)-240);
 
 initmtx_p = rand(10,length(vocabulary_p)); 
 initmtx = rand(10,length(vocabulary));                     
 % As we noticed the initial point for pruning or without pruning are
 % different, so it makes sense why sometime the accuracy of pruned data is 
 % lower than the accuracy of data without pruning.
 
 for k=1:10
     initmtx_p(k,:) = initmtx_p(k,:)/norm(initmtx_p(k,:));
     initmtx(k,:) = initmtx(k,:)/norm(initmtx(k,:));
 end
 
 tst = [data1(ridx1(1:40)),data2(ridx2(1:40)),data3(ridx3(1:40))];
 dev = [data1(ridx1(41:80)),data2(ridx2(41:80)),data3(ridx3(41:80))];
 trn = [data1(ridx1(81:end)),data2(ridx2(81:end)),data3(ridx3(81:end))];
 
 tstdata = tst(randomizetst);
 devdata = dev(randomizedev);
 trndata = trn(randomizetrn);
 
 data = [tstdata,devdata,trndata];
 
 for n=1:length(data)
 	 O = data(n).count;
     O = O(P_Vocidx);
     data(n).P_count = O;                           % build a counter for pruned vocab
 end
  
 ndocs = length(data); 
 nvocs_p = length(vocabulary_p);
 category = zeros(ndocs,1);
 tfmtx_p =  zeros(length(vocabulary_p),length(data));
 nvocs = length(vocabulary);
 tfmtx =  zeros(length(vocabulary),length(data));

 for    k = 1:ndocs                                 % build the TF matrix and category vector
        tfmtx_p(:,k) = data(k).P_count;             % after pruning
        tfmtx(:,k) = data(k).count;                 % before pruning
           switch data(k).cate
            case 'Bug Fix', category(k) = 1;
            case 'New Feature', category(k) = 2;
            case 'Enhancement', category(k) = 3;
           end
 end
  
 idfvt_p = log(ndocs./full(sum(tfmtx_p>0,2)));      % build IDF vector with pruning
 tfidf_p = sparse(nvocs_p,ndocs);
 idfvt = log(ndocs./full(sum(tfmtx>0,2)));          % build IDF vector
 tfidf = sparse(nvocs,ndocs);

 for k=1:ndocs                                      % construct TF-IDF matrix
    tfidf_p(:,k) = tfmtx_p(:,k).*idfvt_p;           % after pruning
    tfidf_p(:,k) = tfidf_p(:,k)/norm(tfidf_p(:,k)); 
    tfidf(:,k) = tfmtx(:,k).*idfvt;                 % before pruning
    tfidf(:,k) = tfidf(:,k)/norm(tfidf(:,k));
 end
 
 tstmtx_p = tfidf_p(:,1:120); tstcat = category(1:120); 
 tstmtx = tfidf(:,1:120); tstcat = category(1:120); 
 devmtx_p = tfidf_p(:,121:240); devcat = category(121:240); 
 devmtx = tfidf(:,121:240); devcat = category(121:240); 
 trnmtx_p = tfidf_p(:,241:end); trncat = category(241:end); 
 trnmtx = tfidf(:,241:end); trncat = category(241:end); 

 nclusts = 3;                                            % define the number of clusters we want
 dst = 'cosine';                                         % define the distance we want to use
 initc_p = initmtx_p(1:nclusts,:);                       % define the initial set of centroids
 initc = initmtx(1:nclusts,:);
 % applies the k-means clustering algorithm to the test set
 [idxs_p,centroids_p] = kmeans(tstmtx_p',nclusts,'Distance',dst,'Start',initc_p);
 [idxs,centroids] = kmeans(tstmtx',nclusts,'Distance',dst,'Start',initc);
 % calculating the distance between the cluster center and category center
 % with or without pruning
 for k=1:nclusts
    for n=1:3
        catcentroid_p = sum(tstmtx_p(:,tstcat==n)',1)/sum(tstcat==n);
        catcentroid_p = catcentroid_p/norm(catcentroid_p);
        clucentroid_p = centroids_p(k,:)/norm(centroids_p(k,:));
        clu2catdist_p(n,k) = 1-(catcentroid_p*clucentroid_p');  
        catcentroid = sum(tstmtx(:,tstcat==n)',1)/sum(tstcat==n);
        catcentroid = catcentroid/norm(catcentroid);
        clucentroid = centroids(k,:)/norm(centroids(k,:));
        clu2catdist(n,k) = 1-(catcentroid*clucentroid');
    end
 end
 
 % selection of best cluster-to-category assignment with or without pruning
 permutations_p = perms([1,2,3]);                       % considers all possible assignments
 permutations = perms([1,2,3]); 
 for    k=1:size(permutations_p,1)                      % computes overall distances for all cases
        dist1_p = clu2catdist_p(1,permutations_p(k,1));
        dist2_p = clu2catdist_p(2,permutations_p(k,2));
        dist3_p = clu2catdist_p(3,permutations_p(k,3));
        overalldist_p(k) = dist1_p+dist2_p+dist3_p;
        dist1 = clu2catdist(1,permutations(k,1));
        dist2 = clu2catdist(2,permutations(k,2));
        dist3 = clu2catdist(3,permutations(k,3));
        overalldist(k) = dist1+dist2+dist3;        
 end
 [void,best_p] = min(overalldist_p);                    % gets the best assignment
 [void,best] = min(overalldist);

 figure(1)
 subplot(3,3,m);
 plot(tstcat+randn(size(tstcat))/10,idxs_p+randn(size(idxs_p))/10,'*','markersize',4);
 xlabel('Actual Category Indexes'); ylabel('Cluster Indexes');

 figure(2)
 subplot(3,3,m);
 plot(tstcat+randn(size(tstcat))/10,idxs+randn(size(idxs))/10,'r*','markersize',4);
 xlabel('Actual Category Indexes'); ylabel('Cluster Indexes');

 newidxs_p = zeros(size(idxs_p));
 newidxs = zeros(size(idxs));
 for k=1:3
     newidxs_p(idxs_p==permutations_p(best_p,k)) = k; 
     newidxs(idxs==permutations(best,k)) = k;
 end

 accuracy_p = sum(newidxs_p==tstcat)/length(tstcat)*100;
 fprintf('In %f time iteration, the final accuracy for our testing group with pruning is:%4.2f \n',m,accuracy_p)
 acc_p(m,1) = accuracy_p; 
 accuracy = sum(newidxs==tstcat)/length(tstcat)*100;
 fprintf('In %f time iteration, the final accuracy for our testing group is:%4.2f\n',m,accuracy)
 acc(m,1) = accuracy; 
end
 
acc_t_p = sum(acc_p)/9;
fprintf('After 9 times iteration, the final accuracy for our testing group with pruning is:%4.2f\n',acc_t_p)

acc_t = sum(acc)/9;
fprintf('After 9 times iteration, the final accuracy for our testing group is:%4.2f\n',acc_t)




 