f = '/Users/olivercase/Desktop/CFA/replication pack/provenance/raw/channelLevelRbTimeseriesChestSorted_record_4.mat';
S = load(f);
c = S.channelLevelRbTimeseriesChestSorted;
fprintf('cell size: [%s]\n', num2str(size(c)));
for i = [1 2]
    v = c{i};
    fprintf('\n=== cell{%d}: class=%s size=[%s] ===\n', i, class(v), num2str(size(v)));
    if istimetable(v) || istable(v)
        fprintf('  vars: %s\n', strjoin(string(v.Properties.VariableNames), ', '));
        for k = 1:width(v)
            col = v.(k);
            fprintf('   %-22s %-10s [%s]\n', v.Properties.VariableNames{k}, class(col), num2str(size(col)));
        end
        if istimetable(v)
            tt = v.Properties.RowTimes;
            dt = seconds(tt(2)-tt(1));
            fprintf('  n=%d  dt=%.8g s  fs=%.4f Hz  dur=%.1f s\n', height(v), dt, 1/dt, seconds(tt(end)-tt(1)));
        end
        disp(head(v,3));
    end
end
