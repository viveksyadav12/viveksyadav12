function [K, F] = assemble_system(mesh, K, F, source_func)
% ASSEMBLE_SYSTEM Assembles global stiffness matrix and load vector
%
% Inputs:
%   mesh - mesh structure from generate_mesh_2d
%   K - initialized global stiffness matrix (sparse)
%   F - initialized global load vector
%   source_func - source term function handle: f(x,y)
%
% Outputs:
%   K - assembled global stiffness matrix
%   F - assembled global load vector

    % Loop over all elements
    for e = 1:mesh.n_elements
        % Get element nodes
        elem_nodes = mesh.elements(e, :);

        % Get nodal coordinates
        x = mesh.nodes(elem_nodes, 1);
        y = mesh.nodes(elem_nodes, 2);

        % Compute element stiffness matrix
        Ke = element_stiffness(x, y);

        % Compute element load vector
        Fe = element_load(x, y, source_func);

        % Assemble into global system
        for i = 1:3
            gi = elem_nodes(i);  % global node index
            F(gi) = F(gi) + Fe(i);

            for j = 1:3
                gj = elem_nodes(j);  % global node index
                K(gi, gj) = K(gi, gj) + Ke(i, j);
            end
        end
    end
end


function Ke = element_stiffness(x, y)
% ELEMENT_STIFFNESS Computes element stiffness matrix for linear triangle
%
% For linear triangular elements with nodes (x1,y1), (x2,y2), (x3,y3)

    % Compute area of triangle
    Area = 0.5 * abs((x(2)-x(1))*(y(3)-y(1)) - (x(3)-x(1))*(y(2)-y(1)));

    % Shape function gradients (constant for linear triangles)
    % ∇N1 = [b1, c1]/(2*Area), where b1 = y2-y3, c1 = x3-x2
    b = [y(2)-y(3); y(3)-y(1); y(1)-y(2)];
    c = [x(3)-x(2); x(1)-x(3); x(2)-x(1)];

    % Element stiffness matrix: Ke_ij = ∫ ∇Ni · ∇Nj dA
    % For constant gradients: Ke_ij = (b_i*b_j + c_i*c_j)/(4*Area)
    Ke = zeros(3, 3);
    for i = 1:3
        for j = 1:3
            Ke(i,j) = (b(i)*b(j) + c(i)*c(j)) / (4*Area);
        end
    end
end


function Fe = element_load(x, y, source_func)
% ELEMENT_LOAD Computes element load vector for linear triangle
%
% Inputs:
%   x, y - nodal coordinates [3x1]
%   source_func - source term function handle

    % Compute area of triangle
    Area = 0.5 * abs((x(2)-x(1))*(y(3)-y(1)) - (x(3)-x(1))*(y(2)-y(1)));

    % Centroid of triangle
    xc = mean(x);
    yc = mean(y);

    % Evaluate source term at centroid
    f_val = source_func(xc, yc);

    % For constant source term approximation:
    % Fe_i = ∫ Ni * f dA ≈ f(centroid) * Area/3
    Fe = (f_val * Area / 3) * ones(3, 1);
end
