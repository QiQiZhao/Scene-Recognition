function [image_feats,image_feats2] = TinyImages(image_paths)
N = size(image_paths, 1);  %get the number of images
image_feats = zeros(N, 256);
image_feats2=zeros(16,16,1,N);
for ii = 1:N    
    image = imread(image_paths{ii});         %read images
    resized = imresize(image, [16 16]);       %change the size of images
    image_feats2(:,:,1,ii)=resized;
    image_feats(ii,:) = reshape(resized, 1,256);    %make the data of a image into a vector
    image_feats(ii,:) = image_feats(ii,:) - mean(image_feats(ii,:));   % normalizing them
    image_feats(ii,:) = image_feats(ii,:)./norm(image_feats(ii,:));
end