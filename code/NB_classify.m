function predicted_categories = NB_classify(train_image_feats, ...
    train_labels, test_image_feats, categories)
%get the number of labels,training data and testing data.
cat_N = length(categories);          
train_N = size(train_labels, 1);
test_N = size(test_image_feats, 1);

categories_nums = linspace(1, 15, 15);
predicted_categories = cell(test_N, 1);
train_labels_nums = labels_numbers(train_labels, categories);
scores = zeros(test_N, cat_N);

% loop on all categories
for i=1:cat_N
    
    % get the positive and negative images of every classifier, positives are the ones assigned to
    % the current category and the negatives are the ones not.
    idx_p = find(train_labels_nums == categories_nums(i));
    idx_n = find(train_labels_nums ~=(categories_nums(i)));
    
    % construct the y which is an array contains either 1 or -1 according to the previous indexes
    y = zeros(train_N, 1);
    for j = 1:length(idx_p)
        y(idx_p(j)) = 1;
    end
    for j = 1:length(idx_n)
        y(idx_n(j)) = -1;
    end
    
    % train the classifier
    nb=fitcnb(train_image_feats,y);
    % get the score
    scores(:,i) = posterior(nb,test_image_feats); 
end
% set the test_image to the category which has the biggest score
for i=1:test_N
    
   
    score = scores(i,:);
    [~, idx] = ismember(max(score),score);    
    predicted_categories{i} = categories{idx};
    
end

end