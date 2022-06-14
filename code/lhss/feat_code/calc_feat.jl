
push!(LOAD_PATH, "./")

import Pkg;
Pkg.add("MATLAB")

using LinearAlgebra

function ComputeFaceArea(p1, p2, p3)
    @assert length(p1) == 3 && length(p2) == 3 && length(p3) == 3    
    p1 = vec(p1)
    p2 = vec(p2)
    p3 = vec(p3)
    A = cross(p2 - p1, p3 - p1)
    area = 0.5 * norm(A)
end

function SamplePointsK(vertices, faces, flabel, normalf, densityProposal)    
    points = []
    normals = []
    plabel = []
    count = 0
    for f in faces
        count = count+1
        
        p1 = vertices[f[1]]
        p2 = vertices[f[2]]
        p3 = vertices[f[3]]

        area = ComputeFaceArea(p1, p2, p3)
        expectedSampleNum = area * densityProposal        
        actualSampleNum = round(Int, expectedSampleNum)
        epsilon = rand()
        if epsilon <= expectedSampleNum - actualSampleNum
            actualSampleNum += 1
        end
        
        # Uniform sampling in a triangle.
        for j = 1:actualSampleNum            
            # Generate uniform (alpha, beta).
            alpha = sqrt(rand())
            beta = rand()

            # Barycentric coordinates.
            newPoint = p1 * (1-alpha) + p2 * alpha*(1-beta) + p3 * beta*alpha
            push!(points, newPoint)
            push!(normals,normalf[count])
            push!(plabel,flabel[count])
        end
    end
    retPoints = zeros(length(points), 3)
    retNormals = zeros(length(normals),3)
    for i = 1:length(points)
        retPoints[i, :] = points[i]
        retNormals[i, :] = normals[i]
    end

    retPoints, retNormals, plabel
end

######## This cell extracts per face feature ########

using JSON

input = ARGS[1]
output = ARGS[2]

j = JSON.parsefile(input)

verts = convert(Vector{Vector{Float64}}, j["verts"])
faces = convert(Vector{Vector{Int}}, j["faces"])
normalf = convert(Vector{Vector{Float64}}, j["normals"])
flabel = convert(Vector{Int}, j["groups"])

using MATLAB

mat"""
addpath('lhss/feat_code/extract_face_feat');
addpath(genpath('lhss/feat_code/ExternalTools/gptoolbox'));
addpath(genpath('lhss/feat_code/ExternalTools/toolbox_graph/toolbox_graph'));
"""
    
pts1, ptn1, plb1 = SamplePointsK(verts, faces, flabel, normalf, 15000)    
pts2, ptn2, plb2 = SamplePointsK(verts, faces, flabel, normalf, 4000)
pts3, ptn3, plb3 = SamplePointsK(verts, faces, flabel, normalf, 800)

mat"""
ptsdense.pts = $pts1; ptsdense.ptn = $ptn1; ptsdense.plb = cell2mat($plb1);
ptsmedium.pts = $pts2; ptsmedium.ptn = $ptn2; ptsmedium.plb = cell2mat($plb2);
ptssparse.pts = $pts3; ptssparse.ptn = $ptn3; ptssparse.plb = cell2mat($plb3);
[curv, locpca, locvar] = curv_pca_var(ptssparse,ptsmedium,ptsdense,[1 1 2 2 3 3],[50 100 50 100 50 100]);
spinfea = spin(ptssparse,ptsmedium,3);
[sc, dhist] = shapecontext(ptssparse,ptsmedium);
feas = [];
feas.curv = single(curv); feas.locpca = single(locpca); feas.locvar = single(locvar);
feas.spinfea = single(spinfea); feas.sc = single(sc); feas.dhist = single(dhist);
feas.pts = single($pts2); feas.ptn = single($ptn2); feas.plb = single(cell2mat($plb2));

save(fullfile($output), 'feas');
"""