
function [train_image_paths, train_labels] =get_image_paths(data_path, categories, num_train_per_cat)

num_categories = length(categories); 

train_image_paths = cell(num_categories * num_train_per_cat, 1);
train_labels = cell(num_categories * num_train_per_cat, 1);

for i=1:num_categories
   images = dir( fullfile(data_path, 'training', categories{i}, '*.jpg'));
   for j=1:num_train_per_cat
       train_image_paths{(i-1)*num_train_per_cat + j} = fullfile(data_path, 'training', categories{i}, images(j).name);
       train_labels{(i-1)*num_train_per_cat + j} = categories{i};
   end
   
  
end