%% Cross-validation script
setup_params;
files = dir(DATA_CV_DIR);
colors = 'ymcrgbk';
k=3;

figure
subplot(1,2,1)
for i=1:size(files)
    file = files(i);
    if strcmp(file.name,'.') == 1 || strcmp(file.name,'..') == 1
        continue;
    end
    load(sprintf('%s/%s',DATA_CV_DIR,file.name));
    fprintf('%s\n',file.name);
    color = colors(i);
    for q=1:k
        r = result(q);
        if ~exist('b','var')
            load(sprintf('%s/%s',DATA_DIR,r.data_file));
            n = size(b,1);
        end
        times = cumsum(r.times);

        b_train = b(r.indices~=q);
        A_train = A(r.indices~=q,:);
        b_holdout = b(r.indices==q);
        A_holdout = A(r.indices==q,:);

        delta_train = A_train * r.hist_train - repmat(b_train,1,size(r.hist,2));
        delta = A * r.hist_train - repmat(b,1,size(r.hist,2));
        delta_holdout = A_holdout * r.hist_train - ...
            repmat(b_holdout,1,size(r.hist,2));
        error_hist_train = diag(sqrt(delta_train'*delta_train));
        error_hist = diag(sqrt(delta'*delta));
        error_hist_holdout = diag(sqrt(delta_holdout'*delta_holdout));
        
        % Test
        %record(q).error_train = norm(A_train * x_train - b_train);
        %record(q).error_holdout = norm(A_holdout * x_train - b_holdout);
        %record(q).error = norm(A * x_train - b);
        
        subplot(1,2,1)
        filename = strrep(file.name, '_', ' ');
        if q==1
            plot(log(times),sum(abs(delta_holdout),1)/n,color,...
                'DisplayName',filename(1:end-4))
        else
            plot(log(times),sum(abs(delta_holdout),1)/n,color,...
                'HandleVisibility','off')
        end
        hold on;
        subplot(1,2,2)
        plot(log(times),error_hist_holdout/n,color,...
            'HandleVisibility','off')
%         hold on;
%         plot(times,error_hist_train,'b')
%         plot(times,error_hist,'r')
        hold on;
    end
end
subplot(1,2,1)
ylabel(sprintf('%d-fold CV holdout error (L1)',k))
xlabel('CPU time (minutes)')
legend('toggle')
subplot(1,2,2)
title('')
ylabel(sprintf('%d-fold CV holdout error (L2)',k))
xlabel('CPU time (minutes)')


%% load data
load('data/smaller_data.mat')
n = size(b,1);
% hashmap with stored errors
load('data/cv_results_smaller_data');
k = 10;

j = 1;
for key=results.keys
    fprintf('%s\n',key{1});
    color = colors(j);
    r = results(key{1});
    for i=1:10
        if i==1
            plot(cumsum(r(i).times),r(i).error_hist/n,color, ...
                'DisplayName',key{1});
        else
            plot(cumsum(r(i).times),r(i).error_hist/n,color, ...
                'HandleVisibility','off');            
        end
        hold on;
    end
    j=j+1;
end
ylabel('10-fold CV error')
xlabel('CPU time (minutes)')
legend('toggle')

%% 
figure
j = 1;
for key=results.keys
    fprintf('%s\n',key{1});
    color = colors(j);
    r = results(key{1});
    timeCPU_total = 0;
    error_total = 0;
    for i=1:k
        timeCPU_total = timeCPU_total + r(i).timeCPU;
        error_total = error_total + r(i).error;
    end
    timeCPU_avg = timeCPU_total/k
    error_avg = error_total/k
    plot(timeCPU_avg/60,error_avg/n,color,'Marker','o','DisplayName',key{1});
    hold on;
    j=j+1;
end
ylabel('10-fold CV error')
xlabel('CPU time (minutes)')
legend('toggle')