

function predicted_categories = knn(train_image_feats, train_labels, test_image_feats)

M = size(test_image_feats,1);
k = 20;
distances = vl_alldist2(train_image_feats', test_image_feats');% compute the distance
all_labels = unique(train_labels);
n_labels = size(all_labels, 1);
[~, indices] = sort(distances, 1);
count_labels = zeros(n_labels, M);
for ii = 1:M
    for jj = 1:n_labels
        top_k_labels = train_labels(indices(1:k, ii));
        count_labels(jj,ii) = sum(strcmp(all_labels(jj), top_k_labels));
    end
end
[~, label_indices] = max(count_labels,[],1);
predicted_categories = all_labels(label_indices);
end