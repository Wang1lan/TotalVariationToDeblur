clc;clear;close all;

%{
 总变分正则化，
采用梯度下降求解
L(u) = 1/2 * ||k*u - f||2_.^2 + lambda*TV，TV= x和y方向的梯度平方，和，再开根号
%}

%% 读取模糊图像以及模糊核
original_image = imread("rgbpicture_2.jpg");
original_image = im2double(original_image(:,:,1));

numGPUs = gpuDeviceCount;% 检测GPU数量
if numGPUs > 0
    original_image = gpuArray(original_image);
end

h = im2double(imread('5de_100sp_10as_30Xc.png'));
% blur = conv2(original_image, h, "same"); 
% 由于卷积核为51×51，过大，利用conv2卷积，因为进行补零填充操作，结果图会形成周围的黑边，故用imfilter函数
blur = imfilter(original_image, h, "symmetric", "same", "conv"); % symmetric：对称填充
% blur = blur ./ max(blur(:));
%{ 
关于上行归一化的代码的一些发现：

1、在调用matlab函数进行加噪声或者滤波前，一定要将图像进行归一化，避免后续出现差错，
但，能自己构建噪声或滤波，尽量自己构建，38-39行是自己构建的，41-42行是调用imnoise，
会发现33的输出结果将输入图像归一化了，从而导致迭代出的效果极差，

2、在进行迭代前，将输入图像进行归一化，虽然数据量减少了，函数达到最小值的时间会缩短
（pic_3，仅迭代150次左右，就能达到最小值，但后续进行迭代，函数值会开始增大），
但这样迭代出的图像，效果并没有原图像好，目前的猜测是迭代不够充分
相反，将未归一化的图像进行输入，到第500次迭代时，虽下降速率变缓慢，但仍未达到最小值，
且迭代出的效果明显好于归一化的结果。
%}

noise = 0.01*randn(size(original_image)); % 方差为0.01的噪声
blur = blur + noise; 

% blur = blur ./ max(blur(:)); % 是否是归一化后，噪声所占的比重会增加？
% blur = imnoise(blur, "gaussian", 0, 0.01);% 方差为0.01的噪声，方式二
%{
figure,imshow(original_image, []);
figure,imshow(blur,[]);
%}

%% 模型构建
lambda = 1e-2;
A = @(u) imfilter(u, h, "symmetric", "same", "conv");
AT = @(u) imfilter(u, rot90(h,2), "symmetric", "same", "conv"); % 卷积矩阵的转置
Dx = @(u) [diff(u, 1, 2), u(:,1) - u(:,end)]; % x方向的梯度,列之间做差
Dy = @(u) [diff(u, 1, 1); u(1,:) - u(end,:)];
Ddeno = @(u) sqrt(Dx(u).^2 + Dy(u).^2 + eps);
D = @(u, f) AT(A(u) - f) + lambda*(Dx(Dx(u) ./ Ddeno(u)) + Dy(Dy(u) ./ Ddeno(u))); % L(u)梯度
L2 = @(u) power(norm(u, 'fro'),2);
L = @(u,f) 1/2*L2(A(u)-f) + lambda*sum(sum(sqrt(Dx(u).^2 + Dy(u).^2)));% L(u)

%% 迭代求解
tau = 1e-3; % 迭代步长，控制收敛速度
maxiter = 800;
u = blur; % 初始化
U = zeros(maxiter, 1);
L_u = zeros(maxiter, 1);

for i = 1:maxiter

    unext = u - tau * D(u, blur);% 梯度下降
    
    U(i) = norm(unext - u, 2);
    L_u(i) = L(u, blur);

    if U(i) < 1e-5
        break;
    end
    u = unext;
    figure(11111), 
    subplot(131),imshow(u,[]), title(['迭代次数 ', num2str(i), ' / ', num2str(maxiter)]);
    subplot(132),semilogy(U, '*-'),xlim([1, maxiter]), title('norm(unext - u, 2)');xlabel('iteration');grid on;grid minor;
    subplot(133),semilogy(L_u, '*-'),xlim([1, maxiter]), title('L(u)');xlabel('iteration');grid on;grid minor;
    drawnow();

end

% original_image = gather(original_image); % 将数据传回CPU
% blur = gather(blur);
% u = gather(u);
blur = blur ./ max(blur(:));
u = u ./ max(u(:));

%% 结果评判
PS1 = psnr(original_image, blur);
PS2 = psnr(original_image, u);
SS1 = ssim(original_image, blur);
SS2 = ssim(original_image, u);
rmse1 = sqrt(immse(original_image, blur));
rmse2 = sqrt(immse(original_image, u));

disp(['原图与模糊图', '---', '原图和复原图']);
disp([PS1, PS2]);
disp([SS1, SS2]);
disp([rmse1, rmse2]);












