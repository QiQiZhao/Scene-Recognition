function vocab = BuildVocabularyOfPatch( image_paths, vocab_size, patch_size, patch_stride )
M = patch_size;
% list of the patch features of all images
patch_feats = zeros(M * M, 1000*1500*2);
row = 1;
N = length(image_paths);
    % loop on all the images to get the patch features
    for i=1:N
        disp(i);
        % get image
        img = imread(image_paths{i});
        [img_h, img_w] = size(img);
        y = 1;
        x = 1;
        % get the patch features and accumulate them to the list
        while (y <= img_h - stride)
            while (x <= img_w - stride)
                % extract patch
                img_patch_feats = img(y:y+M-1, x:x+M-1);
                % convet to 1D array
                img_patch_feats = reshape(img_patch_feats',1,[]);
                patch_feats(:, row) = img_patch_feats';
                row = row + 1;
                x = x + patch_stride;
            end
            x = 1;
            y = y + patch_stride;
        end
    end
    
 % resize and save patch features
patch_feats = patch_feats(:, 1:row-1);
patch_feats = single(patch_feats);
K = vocab_size;
% after calculating patch,using k-means to cluster them into k clusters
[C, ~] = vl_kmeans(patch_feats, K, 'distance', 'l1', 'algorithm', 'elkan');
vocab = C;

end