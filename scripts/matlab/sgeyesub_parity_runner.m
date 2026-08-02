function sgeyesub_parity_runner(input_path, output_path, official_root)
%SGEYESUB_PARITY_RUNNER Run the frozen official algorithm on one parity fixture.
%
% The fixture is produced separately by a Slurm CPU export stage.  It contains
% the exact block-1 support and block-2 query arrays consumed by the Python
% port, in the same channel order.  Query annotations and outcomes are
% forbidden.  This runner writes only the official corrected query and compact
% contract metadata; numerical comparison is performed after both outputs are
% frozen.

expected_commit = '2c95b4f46f37670d25399ac0fdd705ae18248b25';
fixture_root = ...
    '/projects/EEG-foundation-model/derived/denoiseNet/sgeyesub/matlab_parity/fixtures';
official_output_root = ...
    '/projects/EEG-foundation-model/derived/denoiseNet/sgeyesub/matlab_parity/official_outputs';
if nargin ~= 3
    error('CGDR:ParityArguments', ...
        'Expected input_path, output_path and official_root.');
end
input_path = char(string(input_path));
output_path = char(string(output_path));
official_root = char(string(official_root));
if isempty(input_path) || isempty(output_path) || isempty(official_root)
    error('CGDR:ParityArguments', 'Parity paths must be non-empty.');
end
if ~strncmp(input_path, [fixture_root filesep], numel(fixture_root) + 1) ...
        || ~strncmp(output_path, [official_output_root filesep], ...
            numel(official_output_root) + 1)
    error('CGDR:DataPlane', ...
        'Parity fixtures and outputs must remain in the registered derived-data roots.');
end
if ~isfile(input_path) || isfile(output_path) || ~isfolder(fileparts(output_path))
    error('CGDR:ParityPaths', ...
        'Input must exist; output must be new and its parent must already exist.');
end
if official_root(1) ~= '/' || isempty(regexp(official_root, '^[A-Za-z0-9_./-]+$', 'once'))
    error('CGDR:OfficialRoot', 'official_root must be a safe absolute path.');
end

required_source = fullfile(official_root, 'algorithms', 'sgeyesub.m');
if ~isfile(required_source)
    error('CGDR:OfficialSourceMissing', ...
        'The frozen official algorithms/sgeyesub.m is unavailable.');
end
[git_status, git_output] = system(sprintf( ...
    'git -C "%s" rev-parse --verify HEAD 2>/dev/null', official_root));
observed_commit = strtrim(git_output);
if git_status ~= 0 || ~strcmp(observed_commit, expected_commit)
    error('CGDR:OfficialCommit', 'Official checkout is not at the frozen commit.');
end

fixture = load(input_path);
required_fields = {
    'protocol_id', 'study', 'participant_stem', 'layout_id', ...
    'official_source_commit', 'support_block', 'query_block', ...
    'support_data', 'support_artifactclasses', 'query_data', ...
    'support_channel_labels', 'support_channel_types', ...
    'query_channel_labels', 'query_channel_types', 'eeg_chan_idxs_matlab'
};
for index = 1:numel(required_fields)
    if ~isfield(fixture, required_fields{index})
        error('CGDR:FixtureField', ...
            'Parity fixture is missing required field %s.', required_fields{index});
    end
end
forbidden_fields = {
    'query_artifactclasses', 'query_eog', 'query_trial_labels', ...
    'query_trial_ids', 'query_outcomes', 'clean_target'
};
for index = 1:numel(forbidden_fields)
    if isfield(fixture, forbidden_fields{index})
        error('CGDR:QueryLeakage', ...
            'Parity fixture contains forbidden query field %s.', forbidden_fields{index});
    end
end

if ~strcmp(char(string(fixture.protocol_id)), ...
        'sgeyesub_release_internal_block1_to_block2_parity_v1')
    error('CGDR:Protocol', 'Unexpected parity protocol.');
end
if ~strcmp(char(string(fixture.official_source_commit)), expected_commit)
    error('CGDR:FixtureCommit', 'Fixture official commit differs from the frozen commit.');
end
if double(fixture.support_block) ~= 1 || double(fixture.query_block) ~= 2
    error('CGDR:BlockSplit', 'Parity requires block 1 support and block 2 query.');
end
study_value = char(string(fixture.study));
participant_value = char(string(fixture.participant_stem));
layout_value = char(string(fixture.layout_id));
if isempty(regexp(study_value, '^study0[1-5]$', 'once')) ...
        || isempty(regexp(participant_value, ...
            ['^' study_value '_p[0-9]+$'], 'once')) ...
        || isempty(layout_value)
    error('CGDR:ReleaseIdentity', ...
        'Study, release participant stem or layout ID is invalid.');
end

support = double(fixture.support_data);
query = double(fixture.query_data);
if ~ismatrix(support) || ~ismatrix(query) ...
        || size(support, 1) ~= size(query, 1)
    error('CGDR:DataShape', ...
        'Support/query must be channel-by-sample matrices with matching channels.');
end
if any(~isfinite(support(:))) || any(~isfinite(query(:)))
    error('CGDR:DataFinite', 'Support/query data contain NaN or Inf.');
end

support_labels = cellstr(string(fixture.support_channel_labels(:)));
support_types = upper(strtrim(cellstr(string(fixture.support_channel_types(:)))));
query_labels = cellstr(string(fixture.query_channel_labels(:)));
query_types = upper(strtrim(cellstr(string(fixture.query_channel_types(:)))));
if numel(support_labels) ~= size(support, 1) ...
        || numel(support_types) ~= size(support, 1) ...
        || numel(query_labels) ~= size(query, 1) ...
        || numel(query_types) ~= size(query, 1)
    error('CGDR:ChannelMetadata', 'Channel metadata length differs from data.');
end
if numel(unique(support_labels)) ~= numel(support_labels) ...
        || numel(unique(query_labels)) ~= numel(query_labels)
    error('CGDR:ChannelMetadata', 'Channel labels must be unique and ordered.');
end
if ~isequal(support_labels, query_labels) || ~isequal(support_types, query_types)
    error('CGDR:ChannelOrder', ...
        'Block 1 and block 2 channel labels/types/order differ.');
end
official_eeg_indices = find(strcmp(support_types, 'EEG'));
fixture_eeg_indices = double(fixture.eeg_chan_idxs_matlab(:));
if ~isequal(official_eeg_indices, fixture_eeg_indices)
    error('CGDR:EEGChannelRule', ...
        'Fixture EEG indices differ from the official ordered type==EEG rule.');
end

artifactclasses = double(fixture.support_artifactclasses);
if ~isvector(artifactclasses)
    error('CGDR:ArtifactClasses', 'Support artifactclasses must be a vector.');
end
artifactclasses = reshape(artifactclasses, 1, []);
support_sample_count = size(support, 2);
if numel(artifactclasses) ~= support_sample_count
    error('CGDR:ArtifactClasses', ...
        'Support artifactclasses do not align sample-wise with support data.');
end
if any(~isfinite(artifactclasses)) || any(artifactclasses ~= fix(artifactclasses)) ...
        || any(artifactclasses < 0 | artifactclasses > 6)
    error('CGDR:ArtifactClasses', 'Support artifactclasses must be integers 0--6.');
end

addpath(genpath(official_root));
resolved_sgeyesub = which('sgeyesub');
if isempty(resolved_sgeyesub) || ~strcmp(resolved_sgeyesub, required_source)
    error('CGDR:PathIsolation', 'sgeyesub did not resolve inside official_root.');
end

algorithm = sgeyesub();
algorithm.fit(support, artifactclasses, fixture_eeg_indices);
corrected_query = algorithm.apply(query);
if ~isequal(size(corrected_query), size(query)) ...
        || any(~isfinite(corrected_query(:)))
    error('CGDR:OfficialOutput', 'Official correction output is invalid.');
end

protocol_id = char(string(fixture.protocol_id));
study = study_value;
participant_stem = participant_value;
layout_id = layout_value;
support_block = 1;
query_block = 2;
channel_labels = support_labels;
channel_types = support_types;
eeg_chan_idxs_matlab = fixture_eeg_indices;
official_source_commit = expected_commit;
reference_runtime = 'MATLAB_official_commit_2c95b4f';
save(output_path, 'corrected_query', 'protocol_id', 'study', ...
    'participant_stem', 'layout_id', 'support_block', 'query_block', ...
    'channel_labels', 'channel_types', 'eeg_chan_idxs_matlab', ...
    'official_source_commit', 'reference_runtime', '-v7.3');
end
