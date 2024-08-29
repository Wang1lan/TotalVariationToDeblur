clc;clear;close all;

%{
 总变分正则化，
采用共轭梯度求解
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
blur = imfilter(original_image, h, "symmetric", "same", "conv"); % symmetric：对称填充
noise = 0.001*randn(size(original_image)); % 方差为0.001的噪声
blur = blur + noise; 

%% 模型构建
lambda = 1e-3;
A = @(u) imfilter(u, h, "symmetric", "same", "conv");
AT = @(u) imfilter(u, rot90(h,2), "symmetric", "same", "conv");% 卷积矩阵的转置
Dx = @(u) [diff(u, 1, 2), u(:,1) - u(:,end)]; % x方向的梯度,列之间做差
Dy = @(u) [diff(u, 1, 1); u(1,:) - u(end,:)];
Ddeno = @(u) sqrt(Dx(u).^2 + Dy(u).^2 + eps);
DAcg = @(u) AT(A(u)) + lambda*(Dx(Dx(u) ./ Ddeno(u)) + Dy(Dy(u) ./ Ddeno(u))); 
L2 = @(u) power(norm(u, 'fro'),2);
L = @(u,f) 1/2*L2(A(u)-f) - lambda*sum(sum(sqrt(Dx(u).^2 + Dy(u).^2)));% L(u)

%% CG(共轭梯度)
% 初始化
b0 = AT(blur);
x = zeros(size(original_image));
r = DAcg(x) - b0;
p = r;
rho0 = r(:)'*r(:);
maxiter = 600;
L_u = zeros(maxiter, 1);
RHO = zeros(maxiter, 1);

% 迭代优化
for i = 1:maxiter
    omega = DAcg(p);
    alp = rho0 / (p(:)'*omega(:));
    x = x + alp*p;
    r = r - alp*omega;
    rho = r(:)'*r(:);
    RHO(i) = sqrt(rho);
    if RHO(i) < 1e-5
        break;
    end
    p = r + (rho/rho0)*p;
    rho0 = rho;
    L_u(i) = L(x, blur);

    figure(1111),
    subplot(131),imshow(abs(x),[]),title(['迭代次数 ', num2str(i),  ' / ',  num2str(maxiter)]);
    subplot(132),semilogy(RHO, '*-'),xlim([1, maxiter]), title('RHO'),xlabel('iteration');grid on;grid minor;
%     subplot(133),semilogy(L_u, '*-'),xlim([1, maxiter]), title('L(u)'),xlabel('iteration');grid on;grid minor;
    drawnow();             
end

%% 结果评判
blur = blur ./ max(blur(:));
x = abs(x);
x = x ./ max(x(:));

PS1 = psnr(original_image, blur);
PS2 = psnr(original_image, x);
SS1 = ssim(original_image, blur);
SS2 = ssim(original_image, x);
rmse1 = sqrt(immse(original_image, blur));
rmse2 = sqrt(immse(original_image, x));

disp(['原图与模糊图', '---', '原图和复原图']);
disp([PS1, PS2]);
disp([SS1, SS2]);
disp([rmse1, rmse2]);

%{
imwrite(x, ['pic2_iter', num2str(maxiter), '_lambda', num2str(lambda), '.jpg']);
%}






