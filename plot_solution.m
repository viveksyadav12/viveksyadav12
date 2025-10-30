function plot_solution(mesh, u, plot_type)
% PLOT_SOLUTION Visualizes the FEM solution
%
% Inputs:
%   mesh - mesh structure
%   u - solution vector
%   plot_type - (optional) 'surface', 'contour', or 'both' (default: 'both')

    if nargin < 3
        plot_type = 'both';
    end

    % Reshape for plotting
    nx = mesh.nx;
    ny = mesh.ny;
    X = reshape(mesh.nodes(:,1), ny+1, nx+1);
    Y = reshape(mesh.nodes(:,2), ny+1, nx+1);
    U = reshape(u, ny+1, nx+1);

    if strcmp(plot_type, 'surface') || strcmp(plot_type, 'both')
        figure;
        surf(X, Y, U);
        xlabel('x');
        ylabel('y');
        zlabel('u(x,y)');
        title('FEM Solution: Surface Plot');
        colorbar;
        shading interp;
        view(3);
    end

    if strcmp(plot_type, 'contour') || strcmp(plot_type, 'both')
        figure;
        contourf(X, Y, U, 20);
        xlabel('x');
        ylabel('y');
        title('FEM Solution: Contour Plot');
        colorbar;
        axis equal;
        grid on;
    end
end
