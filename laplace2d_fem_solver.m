function [u, mesh] = laplace2d_fem_solver(domain, bc, nx, ny, source_func)
% LAPLACE2D_FEM_SOLVER Solves 2D Laplace equation using Finite Element Method
%
% Solves: -∇²u = f in Ω
%         u = g on ∂Ω (Dirichlet boundary)
%
% Inputs:
%   domain - [xmin, xmax, ymin, ymax] domain boundaries
%   bc - boundary condition function handle: bc(x,y)
%   nx - number of elements in x-direction
%   ny - number of elements in y-direction
%   source_func - (optional) source term function handle: f(x,y)
%                 default: f(x,y) = 0 (homogeneous Laplace)
%
% Outputs:
%   u - solution vector at mesh nodes
%   mesh - structure containing mesh information
%
% Example:
%   domain = [0, 1, 0, 1];
%   bc = @(x,y) sin(pi*x).*sin(pi*y);
%   [u, mesh] = laplace2d_fem_solver(domain, bc, 20, 20);

    % Default source term (homogeneous Laplace equation)
    if nargin < 5
        source_func = @(x,y) 0;
    end

    % Generate mesh
    fprintf('Generating mesh...\n');
    mesh = generate_mesh_2d(domain, nx, ny);

    % Number of nodes
    n_nodes = mesh.n_nodes;

    % Initialize global stiffness matrix and load vector
    K = sparse(n_nodes, n_nodes);
    F = zeros(n_nodes, 1);

    % Assemble global stiffness matrix and load vector
    fprintf('Assembling stiffness matrix...\n');
    [K, F] = assemble_system(mesh, K, F, source_func);

    % Apply Dirichlet boundary conditions
    fprintf('Applying boundary conditions...\n');
    [K, F, bc_nodes] = apply_dirichlet_bc(mesh, K, F, bc);

    % Solve the linear system
    fprintf('Solving linear system...\n');
    u = K \ F;

    fprintf('Solution completed!\n');
end
