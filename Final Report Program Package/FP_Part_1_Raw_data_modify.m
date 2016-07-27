% CS 534 - Machine Learning 
% Final Project_Part 1
% Chenyu Wang ( ID: 932-079-604 )
% Hongyan Yi (ID: 932-430-243 )
% Qun Jing (ID:932-011-106 )

clc
clear

load('Bug.mat');
load('Feature.mat');
load('Enhance.mat');

Bug_c = regexp(Bug,'\w+','match');
Bug_words = [Bug_c{:}];

Feature_c = regexp(Feature,'\w+','match');
Feature_words = [Feature_c{:}];


Enhance_c = regexp(Enhance,'\w+','match');
Enhance_words = [Enhance_c{:}];
 
vocabulary = unique([Bug_words,Feature_words,Enhance_words]);

for n = 1:length(Bug)
    Sing_doc_b = Bug{n,1};
    data1(n).text = Sing_doc_b;
    Sing_doc_b = regexp(Sing_doc_b,'\w+','match');
    data1(n).vocab = Sing_doc_b;
    Bug_count = zeros(length(vocabulary),1);
        for k = 1:length(data1(n).vocab);
            w = data1(n).vocab;
            ww = w(1,k);
            idx = find(strcmp(vocabulary,ww));
            Bug_count(idx,1) = Bug_count(idx,1)+1;
        end
    data1(n).count = Bug_count;
    data1(n).cate = 'Bug Fix';
end
    
for n = 1:length(Feature)
    Sing_doc_f = Feature{n,1};
    data2(n).text = Sing_doc_f;
    Sing_doc_f = regexp(Sing_doc_f,'\w+','match');
    data2(n).vocab = Sing_doc_f;
    Feature_count = zeros(length(vocabulary),1);
        for k = 1:length(data2(n).vocab);
            w = data2(n).vocab;
            ww = w(1,k);
            idx = find(strcmp(vocabulary,ww));
            Feature_count(idx,1) = Feature_count(idx,1)+1;
        end
    data2(n).count = Feature_count;
    data2(n).cate = 'New Feature';
end

for n = 1:length(Enhance)
    Sing_doc_e = Enhance{n,1};
    data3(n).text = Sing_doc_e;
    Sing_doc_e = regexp(Sing_doc_e,'\w+','match');
    data3(n).vocab = Sing_doc_e;
    Enhance_count = zeros(length(vocabulary),1);
        for k = 1:length(data3(n).vocab);
            w = data3(n).vocab;
            ww = w(1,k);
            idx = find(strcmp(vocabulary,ww));
            Enhance_count(idx,1) = Enhance_count(idx,1)+1;
        end
    data3(n).count = Enhance_count;
    data3(n).cate = 'Enhancement';
end
  
 disp('The row data modification is finished.')
  
 