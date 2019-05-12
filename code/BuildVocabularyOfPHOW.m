function vocab = BuildVocabularyOfPHOW( image_paths, vocab_size, phow_steps )
phow_feats = zeros(128, 0);
N = length(image_paths);
% loop on all the images to get the patch features
for i=1:N
        % get cropped image
        img_cropped = get_image_cropped(image_paths{i});
        % convet to single precision for the sake of vl_phow
        img_single = single(img_cropped);
        % get the phow features and accumulate them to the list
        [~, img_phow_feats] = vl_phow(img_single, 'Step', phow_steps);
        phow_feats = [phow_feats img_phow_feats];
end
% after calculating phow,using k-means to cluster them into k clusters
phow_feats = single(phow_feats);
K = vocab_size;
[C, ~] = vl_kmeans(phow_feats, K, 'distance', 'l1', 'algorithm', 'elkan');
vocab = C;
end