% 29 April created by Rob Romijnders
% Information is on my github io at
% robromijnders.github.io

clear all
close all
clc

%Command works only when music_data is in the scope of the current folder
direc_list = dir('music_data');
labels = {'mb','dm'};

num_music = length(direc_list);
names_music = {};
[names_music{1:num_music}] = deal(direc_list.name);


%clear direc_list of any non-music names
i=1;
while i <= num_music
    ind_str = [strfind(names_music{i},'dm') strfind(names_music{i},'mb')];     %Check indices where this string is present
    if isempty(ind_str)                      %Check if that indices are empty
        names_music(i) = [];                      %if so, remove this cell
        num_music = num_music - 1;
    else
        i=i+1;
    end
end

%Butter filter to prvent aliasing
[b,a] = butter(4,1200*2/44100);

%How long must the chunks be?
D = 1999;  %length-1 for Matlab indexing

%Open a csv
dlmwrite('data_music.csv',zeros(1,1000),'delimiter',',');

fss = zeros(num_music,1);
disp('Start to read music files')
tic
line = 0;
for i = 1:num_music
    line_prev = line;
    [m,fs] = audioread(names_music{i});
    ind_str = strfind(names_music{i},'dm');
    if isempty(ind_str)
        label = 0;
    else
        label = 1;
    end    
    %Apply butter filter and downsample
    m = filtfilt(b,a,m);
    m = downsample(m,30);
    
    %Obtain pieces of length D
    len = length(m);
    n = 1;
    while n < len-(D+1)
        chunk = m(n:n+D,1);
        %csvwrite('data_music.csv',chunk',line,0);
        dlmwrite('data_music.csv',[label chunk'],'delimiter',',','-append');
        %disp(sprintf('Chunk starts at %.0f out of %.0f',n,len))
        n = n+D+1;
        line = line+1;
    end
    
    fss(i) = fs;
    disp(sprintf('read file %.0f out of %.0f and reported %.0f lines',i,num_music,line-line_prev))
end
time=toc;
disp(sprintf('Finished reading music files in %.3f seconds with %.0f lines',time,line))

