
function image_feats = GetBagsOfPatch(image_paths, vocab, patch_size, patch_stride)
[~, D] = size(vocab);
stride = patch_stride;
N = size(image_paths, 1);
image_feats = zeros(N,D);

for i=1:N
    disp(i);
    patch_feats = zeros((patch_size^2), 0);
    % get image
    img = imread(image_paths{i});
    [img_h, img_w] = size(img);
    y = 1;
    x = 1;
    % get the patch features and accumulate them to the list
    while (y <= img_h - stride)
        while (x <= img_w - stride)
            % extract patch
            img_patch_feats = img(y:y+patch_size-1, x:x+patch_size-1);
            % convet to 1D array
            img_patch_feats = reshape(img_patch_feats',1,[]);
            patch_feats = [patch_feats img_patch_feats'];
            x = x + stride;
        end
        x = 1;
        y = y + stride;
    end    
    patch_feats = single(patch_feats);
    img_histo = zeros(1, D);
    Q = size(patch_feats, 2);
    distances = zeros(D, 1);
    for j=1:Q
        % for each local feature, get the distances to the centroids
        for k=1:D
            distances(k) = vl_alldist2(patch_feats(:,j), vocab(:,k), 'l1');
        end
        % get the index of the nearest distance
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
