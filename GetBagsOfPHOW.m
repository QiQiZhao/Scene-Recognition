function image_feats = GetBagsOfPHOW(image_paths, phow_steps, vocab)
[~, D] = size(vocab);
N = size(image_paths, 1);
steps = phow_steps;
image_feats = zeros(N,D);
% extract phow features for all the given images
for i=1:N
    % get cropped image
    img_cropped = get_image_cropped(image_paths{i});
    % convet to single precision for the sake of vl_phow
    img_single = single(img_cropped);
    % get the phow features for the image
    [~, img_phow_feats] = vl_phow(img_single, 'Step', steps);
    % single the phow feats for the sake of distance measurement
    img_phow_feats = single(img_phow_feats);
    img_histo = zeros(1, D);
    Q = size(img_phow_feats, 2);
    distances = zeros(D, 1);
    for j=1:Q
        % for each local feature, get the distances to the centroids
        for k=1:D
            distances(k) = vl_alldist2(img_phow_feats(:,j), vocab(:,k), 'l1');
        end
        % then get the index of the nearest centroid 
        [~, index] = ismember(min(distances), distances);
        % increment it to the histogram
        img_histo(index) = img_histo(index) + 1;        
    end
    % normalize the histogram
    img_histo = img_histo/norm(img_histo);    
    % add it to the features matrix
    image_feats(i,:) = img_histo;
end
end