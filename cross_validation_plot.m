%% Cross-validation script

% load data
load('data/smaller_data.mat')
n = size(b,1);
% hashmap with stored errors
load('data/cv_results_smaller_data');
colors = 'ymcrgbk';
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