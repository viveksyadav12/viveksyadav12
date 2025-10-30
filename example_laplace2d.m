% EXAMPLE_LAPLACE2D - Demonstration of 2D Laplace FEM Solver
%
% This script demonstrates various examples of solving the 2D Laplace
% equation using the Finite Element Method with different boundary
% conditions and source terms.

clear all; close all; clc;

%% Example 1: Homogeneous Laplace with sinusoidal BC
% Solves: ∇²u = 0 in [0,1]x[0,1]
%         u = sin(πx)sin(πy) on boundary

fprintf('======================================\n');
fprintf('Example 1: Homogeneous Laplace Equation\n');
fprintf('======================================\n\n');

% Domain definition
domain = [0, 1, 0, 1];

% Boundary condition
bc1 = @(x,y) sin(pi*x).*sin(pi*y);

% Mesh resolution
nx = 30;
ny = 30;

% Solve
[u1, mesh1] = laplace2d_fem_solver(domain, bc1, nx, ny);

% Plot
plot_solution(mesh1, u1, 'both');


%% Example 2: Laplace with polynomial BC
% Solves: ∇²u = 0 in [0,1]x[0,1]
%         u = x² + y² on boundary

fprintf('\n======================================\n');
fprintf('Example 2: Polynomial Boundary Conditions\n');
fprintf('======================================\n\n');

% Boundary condition: u = x² + y²
bc2 = @(x,y) x.^2 + y.^2;

% Solve
[u2, mesh2] = laplace2d_fem_solver(domain, bc2, nx, ny);

% Plot
plot_solution(mesh2, u2, 'both');


%% Example 3: Poisson equation with source term
% Solves: -∇²u = f in [0,1]x[0,1]
%         u = 0 on boundary
%         f = 2π²sin(πx)sin(πy)
% Analytical solution: u = sin(πx)sin(πy)

fprintf('\n======================================\n');
fprintf('Example 3: Poisson Equation with Source\n');
fprintf('======================================\n\n');

% Boundary condition (homogeneous)
bc3 = @(x,y) 0;

% Source term
source3 = @(x,y) 2*pi^2*sin(pi*x).*sin(pi*y);

% Solve
[u3, mesh3] = laplace2d_fem_solver(domain, bc3, nx, ny, source3);

% Analytical solution for comparison
X = reshape(mesh3.nodes(:,1), ny+1, nx+1);
Y = reshape(mesh3.nodes(:,2), ny+1, nx+1);
u_exact = sin(pi*X).*sin(pi*Y);

% Compute error
error = norm(u3(:) - u_exact(:)) / norm(u_exact(:));
fprintf('Relative L2 error: %.6e\n', error);

% Plot
plot_solution(mesh3, u3, 'both');

% Plot analytical solution
figure;
surf(X, Y, u_exact);
xlabel('x');
ylabel('y');
zlabel('u(x,y)');
title('Analytical Solution');
colorbar;
shading interp;


%% Example 4: Heat distribution problem
% Solves: ∇²u = 0 in [-1,1]x[-1,1]
%         u = 100 on top edge (y=1)
%         u = 0 on other edges

fprintf('\n======================================\n');
fprintf('Example 4: Heat Distribution Problem\n');
fprintf('======================================\n\n');

% Domain
domain4 = [-1, 1, -1, 1];

% Boundary condition function
bc4 = @(x,y) bc_heat(x,y);

% Solve with finer mesh
[u4, mesh4] = laplace2d_fem_solver(domain4, bc4, 40, 40);

% Plot
plot_solution(mesh4, u4, 'both');


%% Example 5: Circular boundary condition approximation
% Demonstrates BC that varies with position

fprintf('\n======================================\n');
fprintf('Example 5: Radial Boundary Condition\n');
fprintf('======================================\n\n');

% Boundary condition based on distance from center
bc5 = @(x,y) 100 * exp(-((x-0.5).^2 + (y-0.5).^2));

% Solve
[u5, mesh5] = laplace2d_fem_solver(domain, bc5, nx, ny);

% Plot
plot_solution(mesh5, u5, 'both');

fprintf('\n======================================\n');
fprintf('All examples completed!\n');
fprintf('======================================\n');


%% Helper function for Example 4
function val = bc_heat(x, y)
    % Returns BC value based on position
    tol = 1e-10;
    if abs(y - 1) < tol
        val = 100;  % Hot top edge
    else
        val = 0;    % Cold other edges
    end
end
