import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches, Find2D3DCorrespondences
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):

  # Normalize coordinates (to points on the normalized image plane)
  ks1 = np.concatenate((im1.kps, np.expand_dims(np.ones((im1.kps.shape[0])), axis=1)), axis=1)
  ks2 = np.concatenate((im2.kps, np.expand_dims(np.ones((im2.kps.shape[0])), axis=1)), axis=1)
  normalized_kps1 = np.linalg.inv(K).dot(ks1.T).T
  normalized_kps2 = np.linalg.inv(K).dot(ks2.T).T
  #print(normalized_kps1.shape)
  #print(matches)

  
  # Assemble constraint matrix as equation 2.1
  constraint_matrix = np.ones((matches.shape[0], 9))
  for i in range(matches.shape[0]):
    # Add the constraints
    x_p,y_p,_ = normalized_kps1[matches[i,0]]
    x,y,_ = normalized_kps2[matches[i,1]]
    constraint_matrix[i] = [x_p*x, x_p*y,x_p,y_p*x,y_p*y,y_p,x,y,1]

  
  # Solve for the nullspace of the constraint matrix
  _, s, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]
  #print(vectorized_E_hat.shape, s.shape)

  # Reshape the vectorized matrix to it's proper shape again
  E_hat = vectorized_E_hat.reshape((3,3))
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  U,d,vt = np.linalg.svd(E_hat)
  #print(d)
  E = np.dot(U.dot(np.array([[d[0],0,0],[0,d[1],0],[0,0,0]])),vt)
  #print(E)

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]
    #print(abs(kp1.transpose() @ E @ kp2))

    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  #print(R2,t2, R1, t1)
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)
  #print(K)
  #print(P1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]
    #print(points3D)

  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
  points_3D = np.concatenate([points3D, np.expand_dims(np.ones((points3D.shape[0])), axis=1)], axis = 1)
  #print(points3D)
  #print(np.append(R1, np.expand_dims(t1, 1), 1))

  world_p1 = np.append(np.append(R1, np.expand_dims(t1, 1), 1), np.array([[0,0,0,1]]), 0)
  world_p2 = np.append(np.append(R2, np.expand_dims(t2, 1), 1), np.array([[0,0,0,1]]), 0)
  print(world_p1, world_p2)
  new_pts_1 = world_p1.dot(points_3D.T).T
  new_pts_2 = world_p2.dot(points_3D.T).T
  idxs_1 = []
  idxs_2 = []
  for i in range(points3D.shape[0]):
    if new_pts_1[i,2] >= 0:
      idxs_1.append(i)
    if new_pts_2[i,2] >= 0:
      idxs_2.append(i)
  #print(im1_corrs.shape, im2_corrs.shape)

  common = np.intersect1d(idxs_1, idxs_2)
  if common.shape[0] == 0:
    common = [0,1]

  # Filter points behind the first camera
  im1_corrs = im1_corrs[common]
  im2_corrs = im2_corrs[common]
  points3D = points3D[common]

  # Filter points behind the second camera
  # im1_corrs = im1_corrs[idxs_2]
  # im2_corrs = im2_corrs[idxs_2]
  # points3D = points3D[idxs_2]
  #print(idxs_2)

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  points2D = np.concatenate((points2D, np.expand_dims(np.ones((points2D.shape[0])), axis=1)), axis=1)
  normalized_points2D = np.linalg.inv(K).dot(points2D.T).T
  normalized_points2D = normalized_points2D[:,:-1]

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered image

  
  
  image = images[image_name]
  corrs = {}
  points3D = np.zeros((0,3))
  print(matches)
  for i in registered_images:
    mat = GetPairMatches(image_name, i, matches)
    pts_3d, im1_pts, im2_pts = TriangulatePoints(K, image, images[i], mat)
    corrs[(i, image_name)] = np.concatenate(im1_pts, im2_pts, 1)
    idxs, pts = Find2D3DCorrespondences(image_name, images, matches, registered_images)
    images[i].Add3DCorrs(idxs, pts)
    points3D = np.append(points3D, pts_3d)

  
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  
  
  return points3D, corrs
  
