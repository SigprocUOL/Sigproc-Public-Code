function tdoa = findPathTDOA(node, ref, indices, TDOA_M, visited)
% Check if the node has already been visited
if visited(node)
    tdoa = NaN; % Avoid loops
    return;
end

% Mark the node as visited
visited(node) = true;

% Direct connection case
if any(all(indices == [node, ref], 2))
    idx = find(all(indices == [node, ref], 2));
    tdoa = TDOA_M(indices(idx, 1), indices(idx, 2));
    return;
elseif any(all(indices == [ref, node], 2))
    idx = find(all(indices == [ref, node], 2));
    tdoa = -TDOA_M(indices(idx, 1), indices(idx, 2));
    return;
else
    % Find neighbors of the node in the MST
    neighbors = unique(indices(any(indices == node, 2), :));
    neighbors(neighbors == node) = [];

    % Recursive search among neighbors
    for neighbor = neighbors'
        tdoa_partial = findPathTDOA(neighbor, ref, indices, TDOA_M, visited);
        if ~isnan(tdoa_partial)
            if any(all(indices == [node, neighbor], 2))
                idx = find(all(indices == [node, neighbor], 2));
                tdoa = TDOA_M(indices(idx, 1), indices(idx, 2)) + tdoa_partial;
            elseif any(all(indices == [neighbor, node], 2))
                idx = find(all(indices == [neighbor, node], 2));
                tdoa = -TDOA_M(indices(idx, 1), indices(idx, 2)) + tdoa_partial;
            end
            return;
        end
    end
end

% If no path found, return NaN
tdoa = NaN;
end
