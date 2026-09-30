f = '/Users/olivercase/Desktop/CFA/replication pack/provenance/raw/channelLevelRbTimeseriesChestSorted_record_4.mat';
w = whos('-file', f);
fprintf('=== variables in file ===\n');
for i = 1:numel(w)
    fprintf('%-30s %-12s size=[%s]\n', w(i).name, w(i).class, num2str(w(i).size));
end
m = matfile(f);
for i = 1:numel(w)
    n = w(i).name;
    fprintf('\n=== %s ===\n', n);
    try
        v = m.(n);
        if istimetable(v) || istable(v)
            fprintf('  vars: %s\n', strjoin(string(v.Properties.VariableNames), ', '));
            fprintf('  height=%d width=%d\n', height(v), width(v));
            if istimetable(v)
                tt = v.Properties.RowTimes;
                fprintf('  t0=%s tEnd=%s\n', string(tt(1)), string(tt(end)));
                dt = seconds(tt(2)-tt(1));
                fprintf('  dt=%.6g s  => fs=%.4f Hz\n', dt, 1/dt);
            end
            disp(head(v, 3));
        else
            fprintf('  class=%s size=[%s]\n', class(v), num2str(size(v)));
        end
    catch ME
        fprintf('  <could not read: %s>\n', ME.message);
    end
end
