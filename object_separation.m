src_img = imread("circles.jpg");
gray_img = im2gray(src_img);
BW = imbinarize(gray_img);
BW = ~BW;
imwrite(BW,"binary_inv.jpg");
BW2 = bwmorph(BW,'erode',45);
imwrite(BW,"erosed.jpg");
BW2 = bwmorph(BW2,'thicken',Inf);
imwrite(BW,"boundaries.jpg");
BW = ~(BW & BW2);
imwrite(BW,"result.jpg");