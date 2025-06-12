root = 'images';
files = dir(fullfile(root, '*.png'));
files = sort({files.name})';

%%

read_image = @(file) im2double(imread(fullfile(root, file)));
images = cellfun(read_image, files, 'UniformOutput',false);

%%

file_pairs = pairs_with_replacement(files);
image_pairs = pairs_with_replacement(images);

num_pairs = size(file_pairs, 1);
scores = zeros(num_pairs, 1);
for k = 1 : num_pairs
   
    scores(k) = ssim(image_pairs{k, :});
end

%%

data = table(file_pairs(:, 1), file_pairs(:, 2), scores, ...
    'VariableNames',{'image1', 'image2', 'score'});
writetable(data, 'reference.csv')


%%

function [pairs] = pairs_with_replacement(x)

x = x(:);
num_elements = length(x);
num_pairs = num_elements * (num_elements + 1) / 2;

k = 1;
pairs = cell(num_pairs, 2);
for i = 1 : num_elements
    for j = i : num_elements
        pairs(k, :) = x([i, j]);
        k = k + 1;
    end
end

end

