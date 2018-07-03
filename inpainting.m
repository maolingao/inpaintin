% read the image and the mask from input files
f = double(imread('corrupted.png'));
[ny, nx, c] = size(f);
N=nx*ny*c;
% mask(i,j) = 1 iff pixel (i,j) is corrupted.
mask = imread('mask.png') == zeros(ny, nx, c);

% vectorize image and mask
f=f(:);
mask = mask(:);
%%
% count corrupted pixels
M = sum(mask);

% construct a sparse linear operator for the color gradient
e = ones(ny, 1);
A = spdiags([-e e], 0:1, ny, ny);
A(end, end) = 0;
Dy = kron(speye(nx), A);
e = ones(nx, 1);
B = spdiags([-e e], 0:1, nx, nx)';
B(:, end) = 0;
Dx = kron(B', speye(ny));
Nabla_x = kron(speye(3), Dx); 
Nabla_y = kron(speye(3), Dy);


%%
% Let u_tilde be the vector containing only the unknown pixels that we
% optimize for and let u be the whole image that we are looking for.
% Construct operators X, Y s.t. u=X*u_tilde + Y*f is the reconstructed image.
X = sparse(find(mask), 1:M, 1, N, M);
Y = speye(N);
Y(mask, :) = 0;

% Construct a least squares problem min_{u_tilde} F(u_tilde) 
% with F(u_tilde)=|| A * u_tilde - b ||^2
A1 = Nabla_x * X;
b1 = Nabla_x * Y * f;
A2 = Nabla_y * X;
b2 = Nabla_y * Y * f;

%% solve for u_tilde using least square
u_tilde_ls = (A1'*A1 + A2'*A2) \ (-A1'*b1 - A2'*b2);

%% solve for u_tilde using ADMM
epsilon = 1e-8;
u_tilde = 255*rand(M,1);
z       = 255*rand(M,1);
w       = zeros(M,1);
rho     = 1e0;
i       = 0;
res_p   = 1;
res_d   = 1;
ite_max = 1e3;
tic;
while( res_p > epsilon || res_d > epsilon )
    
	if(i > ite_max) %#ok<ALIGN>
		break;
    end    
	if mod(i, 100) == 0 %#ok<ALIGN>
		fprintf('iteration %4i, primal residual = %8.2e, dual residual = %8.2e \n', i, res_p, res_d);
        u = X*(u_tilde) + Y*f;
        imshow([uint8(reshape(f, ny, nx, 3)) uint8(reshape(u, ny, nx, 3))]);
        title(sprintf('iteration %4d', i));
        drawnow;
		% fflush(stdout);
    end
	u_tilde = (2*(A1'*A1) + rho*speye(M)) \ (-2*A1'*b1 + rho*(z - w));
    z_prev  = z;
	z       = (2*(A2'*A2) + rho*speye(M)) \ (-2*A2'*b2 + rho*(u_tilde + w));
	w       = w + u_tilde - z;
    res_p   = max(abs(u_tilde - z));
    % res_p   = norm(u_tilde - z, 'fro');
    res_d   = max(abs(rho * (z - z_prev)));
	i=i+1;
end
t = toc;
fprintf('===================================================\n');
fprintf('DONE, ADMM found solution in %4d steps (%5.2i seconds).\nThe final primal residual = %8.2e \nThe final dual residual = %8.2e \n', i, t, res_p, res_d);

% Inpaint unknown pixels in f
u_ls = X*u_tilde_ls + Y*f;
u    = X*(u_tilde) + Y*f;
u_z    = X*z + Y*f;
fprintf('||u_admm - u_ls|| = %4.4e \n', norm(u - u_ls, 2));

imshow([uint8(reshape(f, ny, nx, 3)) uint8(reshape(u, ny, nx, 3)) uint8(reshape(u_z, ny, nx, 3)) uint8(reshape(u_ls, ny, nx, 3))]);
title('ADMM & least square')
figure, imshow(uint8(reshape(u, ny, nx, 3)) );
title('ADMM')
figure, imshow(uint8(reshape(u_ls, ny, nx, 3)) );
title('ls')
figure, imshow(uint8(reshape(f, ny, nx, 3)) );
title('f')
