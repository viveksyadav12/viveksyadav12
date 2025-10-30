function [K, F, bc_nodes] = apply_dirichlet_bc(mesh, K, F, bc_func)
% APPLY_DIRICHLET_BC Applies Dirichlet boundary conditions
%
% Inputs:
%   mesh - mesh structure
%   K - global stiffness matrix
%   F - global load vector
%   bc_func - boundary condition function handle: u(x,y) on boundary
%
% Outputs:
%   K - modified stiffness matrix
%   F - modified load vector
%   bc_nodes - boundary node indices

    bc_nodes = mesh.boundary_nodes;

    % Get boundary node coordinates and evaluate BC
    for i = 1:length(bc_nodes)
        node_idx = bc_nodes(i);
        x = mesh.nodes(node_idx, 1);
        y = mesh.nodes(node_idx, 2);

        % Evaluate boundary condition
        u_bc = bc_func(x, y);

        % Apply BC using penalty method or direct elimination
        % Here we use direct elimination:
        % Set row to identity and RHS to boundary value

        % Zero out the row
        K(node_idx, :) = 0;
        % Set diagonal to 1
        K(node_idx, node_idx) = 1;
        % Set RHS to boundary value
        F(node_idx) = u_bc;
    end
end
