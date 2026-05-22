function [TDOA_est_MST,indicesvec] = func_Compute_MST(Reliability,TDOA_est)

%% MST
TDOA_est_MST = TDOA_est;
[x_co, y_co, Weight_Graph] = deal([]);
N = size(Reliability,1);
% Weight_Graph = zeros(N,N,1,size(Reliability,4));

indicesvec = zeros(N-1,2,1,size(Reliability,4));

for ll = 1:size(Reliability,4)
    Weight(:,:,1,ll) = (1./Reliability(:,:,1,ll));%-Reliability(:,:,1,ll);%(1./Reliability(:,:,1,ll)).^(N);
    for ii = 1:N
        for jj = 1:(ii-1)
            if ii > jj
                x_co = [x_co, ii];
                y_co = [y_co, jj];

                Weight_Graph = [Weight_Graph, Weight(ii,jj,1,ll)]; % Weight(ii,jj,1,ll); % 

                % Weight = [Weight, 1/abs(Noise(ii,jj)).^2];%
                % Weight = [Weight, 1/(Noise(ii,jj))];%
                % (rssq(TDOA_M{1}(ii,jj))/rssq(Noise(ii,jj)))];
                %snr(TDOA_M{1}(ii,jj),Noise(ii,jj))];
            end
        end
    end
    Graph = graph(x_co, y_co, Weight_Graph);
    % p = plot(Graph,'EdgeLabel',Graph.Edges.Weight);
    T_Prim = minspantree(Graph,'Method','dense');
    % T_Kruskal = minspantree(Graph,'Method','sparse');

    % t_Tree_est = zeros(N,1);
    [indicesvecP] ...%,indicesvecK] 
        = deal(zeros(N,2));
    % graph to indices
    for ii = 2:N
        indicesvecP(ii,:) = T_Prim.Edges{ii-1,1};
        % indicesvecK(ii,:) = T_Kruskal.Edges{ii-1,1};
    end

    % Estimate TDOAs for each node with respect to the reference node
    for ii = 2:N
        visited = false(N, 1); % Initialize the visited array for each node
        t_Tree_estP(ii, 1) = findPathTDOA(ii, 1, indicesvecP, TDOA_est(:,:,1,ll), visited);
        % t_Tree_estK(ii, 1) = findPathTDOA(ii, 1, indicesvecK, TDOA_est(:,:,1,ll), visited);
    end

    indicesvec(:,:,1,ll) = indicesvecP(2:end,:);

    TDOA_est_MST(:,:,1,ll) = t_Tree_estP - t_Tree_estP';
end

end