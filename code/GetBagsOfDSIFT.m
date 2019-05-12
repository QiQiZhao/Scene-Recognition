function image_feats = GetBagsOfDSIFT(image_paths, sift_steps, vocab)
[~, D] = size(vocab);
N = size(image_paths, 1);
steps = sift_steps;
image_feats = zeros(N,D);
% extract sift features for all the given images
for i=1:N
    % get cropped image
    img_cropped = get_image_cropped(image_paths{i});
    % convet to single precision for the sake of vl_dsift
    img_single = single(img_cropped);
    % get the sift features for the image
    % notice that dense sift is used
    [~, img_sift_feats] = vl_dsift(img_single, 'fast', 'Size', 3, 'Step', steps);
    % single the sift feats for the sake of distance measurement
    img_sift_feats = single(img_sift_feats);
    img_histo = zeros(1, D);
    Q = size(img_sift_feats, 2);
    distances = zeros(D, 1);
    for j=1:Q
        % for each local feature, get the distances to the centroids
        for k=1:D
            distances(k) = vl_alldist2(img_sift_feats(:,j), vocab(:,k), 'l1');
        end
        % then get the index of the nearest centroid 
        [~, index] = ismember(min(distances), distances);
        %increment it to the histogram
        img_histo(index) = img_histo(index) + 1;        
    end
    % normalize the histogram
    img_histo = img_histo/norm(img_histo);    
    % add it to the features matrix
    image_feats(i,:) = img_histo;
end
end
