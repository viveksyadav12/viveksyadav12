% QUICKSTART - Simple example to get started with 2D Laplace FEM Solver
%
% This is the simplest example to demonstrate the FEM solver usage.

clear all; close all; clc;

fprintf('2D Laplace Equation FEM Solver - Quick Start\n');
fprintf('============================================\n\n');

% Define the domain: [xmin, xmax, ymin, ymax]
domain = [0, 1, 0, 1];

% Define boundary condition: u(x,y) on the boundary
% Example: u = sin(πx)*sin(πy)
bc = @(x,y) sin(pi*x).*sin(pi*y);

% Define mesh resolution (number of elements in x and y directions)
nx = 20;  % elements in x-direction
ny = 20;  % elements in y-direction

% Solve the Laplace equation: ∇²u = 0
fprintf('Solving the 2D Laplace equation...\n');
[u, mesh] = laplace2d_fem_solver(domain, bc, nx, ny);

% Visualize the solution
fprintf('Plotting the solution...\n');
plot_solution(mesh, u);

fprintf('\nDone! Check the generated plots.\n');
