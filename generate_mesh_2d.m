function mesh = generate_mesh_2d(domain, nx, ny)
% GENERATE_MESH_2D Generates a structured triangular mesh for 2D domain
%
% Inputs:
%   domain - [xmin, xmax, ymin, ymax] domain boundaries
%   nx - number of elements in x-direction
%   ny - number of elements in y-direction
%
% Outputs:
%   mesh - structure containing:
%          nodes: [n_nodes x 2] nodal coordinates (x, y)
%          elements: [n_elements x 3] element connectivity (triangles)
%          n_nodes: total number of nodes
%          n_elements: total number of elements
%          boundary_nodes: indices of boundary nodes

    xmin = domain(1); xmax = domain(2);
    ymin = domain(3); ymax = domain(4);

    % Generate grid points
    x = linspace(xmin, xmax, nx + 1);
    y = linspace(ymin, ymax, ny + 1);
    [X, Y] = meshgrid(x, y);

    % Create node list
    mesh.nodes = [X(:), Y(:)];
    mesh.n_nodes = size(mesh.nodes, 1);

    % Create element connectivity (2 triangles per rectangle)
    mesh.elements = [];
    for j = 1:ny
        for i = 1:nx
            % Node indices for rectangle
            n1 = (j-1)*(nx+1) + i;
            n2 = n1 + 1;
            n3 = j*(nx+1) + i + 1;
            n4 = n3 - 1;

            % Triangle 1: [n1, n2, n4]
            mesh.elements = [mesh.elements; n1, n2, n4];

            % Triangle 2: [n2, n3, n4]
            mesh.elements = [mesh.elements; n2, n3, n4];
        end
    end

    mesh.n_elements = size(mesh.elements, 1);

    % Identify boundary nodes
    % Bottom edge
    bottom = 1:(nx+1);
    % Top edge
    top = (ny*(nx+1)+1):mesh.n_nodes;
    % Left edge
    left = 1:(nx+1):((ny)*(nx+1)+1);
    % Right edge
    right = (nx+1):(nx+1):mesh.n_nodes;

    mesh.boundary_nodes = unique([bottom, top, left, right]);

    % Store domain info
    mesh.domain = domain;
    mesh.nx = nx;
    mesh.ny = ny;

    fprintf('Mesh generated: %d nodes, %d elements\n', mesh.n_nodes, mesh.n_elements);
end
