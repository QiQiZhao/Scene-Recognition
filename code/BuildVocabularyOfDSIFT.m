function vocab = BuildVocabularyOfDSIFT( image_paths, vocab_size, sift_steps )
sift_feats = zeros(128, 0);
N = length(image_paths);
% Get the sift features
    for i=1:N
        % get cropped image
        img_cropped = get_image_cropped(image_paths{i});
        % convet to single precision for the sake of vl_dsift
        img_single = single(img_cropped);
        % get the sift features and accumulate them to the list
        % notice that dense sift is used
        [~, img_sift_feats] = vl_dsift(img_single, 'fast', 'Size', 3, 'Step', sift_steps);
        sift_feats = [sift_feats img_sift_feats];
    end
% after calculating sift,using k-means to cluster them into k clusters
sift_feats = single(sift_feats);
K = vocab_size;
[C, ~] = vl_kmeans(sift_feats, K, 'distance', 'l1', 'algorithm', 'elkan');
vocab = C;
end